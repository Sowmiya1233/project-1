"""
hallucination_check.py
Stage 7 of the pipeline: Post-hoc Hallucination Check

Responsibilities:
- Take the GenerationResult from generator.py (answer + per-claim citations)
- For EACH claim, re-check it against ONLY the chunk(s) it cites:
    - Is the claim actually entailed by that evidence? (not contradicted,
      not unsupported/fabricated beyond what the evidence says)
- Flag claims that fail this check as likely hallucinations
- Produce a final "clean" answer with unsupported claims removed, plus a
  faithfulness score for the whole answer

This is a second, independent safety net on top of the pre-generation
verifier.py -- it catches cases where the evidence was fine, but the
generator drifted from it while writing the answer.

Usage:
    from hallucination_check import HallucinationChecker
    checker = HallucinationChecker()
    report = checker.check(generation_result)
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"
ENTAILMENT_PASS_THRESHOLD = 0.5  # claim needs at least this entailment score to be kept


@dataclass
class ClaimCheck:
    claim_text: str
    supporting_chunk_indices: List[int]
    entailment_score: float   # 0.0 = contradicted/unsupported, 1.0 = fully entailed
    verdict: str               # "supported" | "unsupported" | "contradicted"
    reasoning: str = ""


@dataclass
class HallucinationReport:
    original_answer: str
    clean_answer: str          # answer with unsupported/contradicted claims removed
    faithfulness_score: float  # fraction of claims that passed (0.0-1.0)
    claim_checks: List[ClaimCheck] = field(default_factory=list)
    flagged_count: int = 0


class HallucinationChecker:
    def __init__(self, model_name: str = MODEL_NAME):
        self.client = Groq()
        self.model_name = model_name

    def _check_claim(self, claim_text: str, cited_chunks: List[Dict]) -> ClaimCheck:
        if not cited_chunks:
            return ClaimCheck(
                claim_text=claim_text,
                supporting_chunk_indices=[],
                entailment_score=0.0,
                verdict="unsupported",
                reasoning="Claim cites no evidence chunks.",
            )

        evidence_block = "\n\n".join(
            f"[Chunk {i}] {c['text']}" for i, c in cited_chunks
        )

        system_prompt = (
            "You are a hallucination-detection module. Given a single claim and the "
            "exact evidence it was supposedly based on, judge whether the evidence "
            "actually entails the claim. Be strict: the claim must be directly "
            "supported, not just plausible or related. "
            "Respond ONLY with valid JSON, no preamble, no markdown fences. "
            "Keep reasoning to 12 words or fewer."
        )

        user_prompt = f"""Claim: "{claim_text}"

Cited evidence:
{evidence_block}

Score entailment 0.0-1.0 (1.0 = evidence directly and fully supports the claim,
0.0 = evidence contradicts or says nothing relevant to the claim).

Return JSON exactly like:
{{"entailment_score": 0.0, "verdict": "supported", "reasoning": "short reason"}}

verdict must be one of: "supported", "unsupported", "contradicted" """

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ClaimCheck(
                claim_text=claim_text,
                supporting_chunk_indices=[i for i, _ in cited_chunks],
                entailment_score=0.0,
                verdict="unsupported",
                reasoning="parse_error: treating as unsupported to be safe",
            )

        return ClaimCheck(
            claim_text=claim_text,
            supporting_chunk_indices=[i for i, _ in cited_chunks],
            entailment_score=float(parsed.get("entailment_score", 0.0)),
            verdict=parsed.get("verdict", "unsupported"),
            reasoning=parsed.get("reasoning", ""),
        )

    def check(self, generation_result) -> HallucinationReport:
        """`generation_result` is a GenerationResult from generator.py"""
        claim_checks = []

        for claim in generation_result.claims:
            cited_chunks = [
                (i, generation_result.used_chunks[i])
                for i in claim.supporting_chunk_indices
                if i in generation_result.used_chunks
            ]
            check = self._check_claim(claim.text, cited_chunks)
            claim_checks.append(check)

        supported_claims = [
            c for c in claim_checks if c.entailment_score >= ENTAILMENT_PASS_THRESHOLD
        ]
        flagged = [c for c in claim_checks if c not in supported_claims]

        faithfulness_score = (
            round(len(supported_claims) / len(claim_checks), 4) if claim_checks else 0.0
        )

        clean_answer = " ".join(c.claim_text for c in supported_claims) or (
            "No claims could be verified against their cited evidence."
        )

        return HallucinationReport(
            original_answer=generation_result.answer_text,
            clean_answer=clean_answer,
            faithfulness_score=faithfulness_score,
            claim_checks=claim_checks,
            flagged_count=len(flagged),
        )


def format_report(report: HallucinationReport) -> str:
    lines = [
        f"Original answer: {report.original_answer}",
        f"Faithfulness score: {report.faithfulness_score} "
        f"({report.flagged_count} claim(s) flagged)",
        f"Clean (verified-only) answer: {report.clean_answer}\n",
        "Per-claim breakdown:",
    ]
    for c in report.claim_checks:
        flag = "OK" if c.entailment_score >= ENTAILMENT_PASS_THRESHOLD else "FLAGGED"
        lines.append(
            f"  [{flag}] \"{c.claim_text}\" -> entailment={c.entailment_score} "
            f"verdict={c.verdict}"
        )
        if c.reasoning:
            lines.append(f"      reasoning: {c.reasoning}")
    return "\n".join(lines)


def main():
    """Quick manual test wiring the full pipeline together, end to end."""
    from retriever import Retriever
    from verifier import Verifier
    from corrector import Corrector
    from generator import Generator

    r = Retriever()
    v = Verifier()
    c = Corrector(retriever=r, verifier=v)
    g = Generator()
    hc = HallucinationChecker()

    query = input("Enter a test query: ")
    outcome = c.run(query)

    if not outcome.final_passed:
        print("\nVerification never passed -- refusing to generate.")
        return

    result = g.generate(outcome.final_query, outcome.final_retrieval)
    report = hc.check(result)

    print()
    print(format_report(report))


if __name__ == "__main__":
    main()
