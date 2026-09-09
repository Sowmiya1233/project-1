"""
verifier.py
Stage 4 of the pipeline: Adaptive Evidence Verification

Responsibilities:
- For each retrieved chunk, score it on:
    - Relevance   (does it address the query?)
    - Sufficiency (does it contain enough info to answer, combined with other chunks?)
    - Consistency (does it agree with the other retrieved chunks?)
- Adapt verification strictness based on the retriever's confidence score:
    - High retrieval confidence  -> lighter check (cheap heuristic pass)
    - Low/medium retrieval confidence -> stricter check (LLM-judged pass)
- Return a verdict: PASS (send to generation) or FAIL (trigger correction loop)

Usage:
    from retriever import Retriever
    from verifier import Verifier

    r = Retriever()
    v = Verifier()

    retrieval = r.retrieve_with_confidence("What causes hallucination in RAG?")
    verdict = v.verify(query="What causes hallucination in RAG?", retrieval=retrieval)
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Literal

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---- Config ----
MODEL_NAME = "openai/gpt-oss-120b"  # Groq's current recommended general-purpose model

# Confidence thresholds that decide how strict verification needs to be.
# These are the "adaptive" part of adaptive evidence verification.
HIGH_CONFIDENCE_THRESHOLD = 0.55   # above this: cheap heuristic check is enough
LOW_CONFIDENCE_THRESHOLD = 0.35    # below this: verification is at its strictest

# How many of the best chunks (by relevance) to average when scoring.
# Not every retrieved chunk needs to be great individually -- the collection just
# needs enough good evidence among the top ones. Prevents irrelevant filler chunks
# (e.g. boilerplate, off-topic passages) from dragging the average down.
TOP_N_FOR_SCORING = 3

# Minimum average per-chunk relevance score (0-1) required to pass at each strictness level
PASS_THRESHOLDS = {
    "light": 0.35,
    "standard": 0.5,
    "strict": 0.65,
}


@dataclass
class ChunkVerdict:
    chunk_index: int
    relevance: float
    sufficiency: float
    consistency: float
    reasoning: str = ""


@dataclass
class VerificationResult:
    passed: bool
    strictness: Literal["light", "standard", "strict"]
    overall_score: float
    chunk_verdicts: List[ChunkVerdict] = field(default_factory=list)
    reason: str = ""


class Verifier:
    def __init__(self, model_name: str = MODEL_NAME):
        self.client = Groq()  # reads GROQ_API_KEY from environment
        self.model_name = model_name

    # ---- Strictness selection (the "adaptive" logic) ----
    def _select_strictness(self, retrieval_confidence: float) -> str:
        if retrieval_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return "light"
        elif retrieval_confidence <= LOW_CONFIDENCE_THRESHOLD:
            return "strict"
        else:
            return "standard"

    # ---- Cheap heuristic check (used for "light" strictness) ----
    def _heuristic_check(self, query: str, chunks: List[Dict]) -> List[ChunkVerdict]:
        """
        Fast, no-LLM-call check based on lexical overlap between query and chunk text.
        Used only when retrieval confidence is already high, to save cost/latency.
        """
        query_terms = set(query.lower().split())
        verdicts = []
        for i, chunk in enumerate(chunks):
            chunk_terms = set(chunk["text"].lower().split())
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            # reuse retriever's similarity as a proxy for relevance/sufficiency here
            relevance = chunk["similarity"]
            sufficiency = min(1.0, chunk["similarity"] + overlap * 0.2)
            verdicts.append(ChunkVerdict(
                chunk_index=i,
                relevance=round(relevance, 3),
                sufficiency=round(sufficiency, 3),
                consistency=1.0,  # assumed consistent at light strictness
                reasoning="heuristic pass (high retrieval confidence)",
            ))
        return verdicts

    # ---- LLM-judged check (used for "standard" and "strict") ----
    def _llm_check(self, query: str, chunks: List[Dict], strictness: str) -> List[ChunkVerdict]:
        chunk_block = "\n\n".join(
            f"[Chunk {i}]\n{c['text']}" for i, c in enumerate(chunks)
        )

        strictness_note = {
            "standard": "Apply normal scrutiny.",
            "strict": "Apply high scrutiny — be skeptical, only give high scores to clearly strong evidence.",
        }[strictness]

        system_prompt = (
            "You are an evidence verification module in a RAG pipeline. "
            "You do not answer the user's question. You only judge the evidence. "
            f"{strictness_note} "
            "Respond ONLY with valid JSON, no preamble, no markdown fences."
        )

        user_prompt = f"""Query: {query}

Retrieved evidence chunks:
{chunk_block}

For EACH chunk, score (0.0 to 1.0):
- relevance: does this chunk actually address the query?
- sufficiency: combined with the other chunks, is there enough info here to answer accurately?
- consistency: does this chunk agree with the other chunks (no contradictions)?

Keep "reasoning" to 10 words or fewer so the full response fits.

Return JSON in exactly this shape:
{{
  "chunk_scores": [
    {{"chunk_index": 0, "relevance": 0.0, "sufficiency": 0.0, "consistency": 0.0, "reasoning": "short reason"}},
    ...
  ]
}}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=3000,
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
            # Response got truncated or malformed. Try to salvage complete chunk
            # entries from the partial JSON before giving up entirely.
            import re
            entries = re.findall(
                r'\{\s*"chunk_index":\s*(\d+),\s*"relevance":\s*([\d.]+),\s*'
                r'"sufficiency":\s*([\d.]+),\s*"consistency":\s*([\d.]+)',
                text,
            )
            if entries:
                print(f"\n--- NOTE: verifier response was truncated; salvaged {len(entries)} of {len(chunks)} chunk scores. ---\n")
                return [
                    ChunkVerdict(int(idx), float(rel), float(suf), float(con), "salvaged from truncated response")
                    for idx, rel, suf, con in entries
                ]
            # Nothing usable found -> fail safe with zeros so correction loop triggers
            print("\n--- DEBUG: verifier JSON parse failed, nothing salvageable. Raw LLM output: ---")
            print(text)
            print("--- END DEBUG ---\n")
            # Fail safe: if the model didn't return clean JSON, treat as low scores
            # so the correction loop gets triggered rather than silently passing bad evidence.
            return [
                ChunkVerdict(i, 0.0, 0.0, 0.0, "parse_error: could not score this chunk")
                for i in range(len(chunks))
            ]

        verdicts = []
        for item in parsed.get("chunk_scores", []):
            verdicts.append(ChunkVerdict(
                chunk_index=item.get("chunk_index", -1),
                relevance=float(item.get("relevance", 0.0)),
                sufficiency=float(item.get("sufficiency", 0.0)),
                consistency=float(item.get("consistency", 0.0)),
                reasoning=item.get("reasoning", ""),
            ))
        return verdicts

    def verify(self, query: str, retrieval: Dict) -> VerificationResult:
        """
        Main entry point. `retrieval` is the dict returned by
        Retriever.retrieve_with_confidence(): {"chunks": [...], "confidence": float}
        """
        chunks = retrieval["chunks"]
        retrieval_confidence = retrieval["confidence"]

        if not chunks:
            return VerificationResult(
                passed=False, strictness="strict", overall_score=0.0,
                reason="No chunks retrieved.",
            )

        strictness = self._select_strictness(retrieval_confidence)

        if strictness == "light":
            verdicts = self._heuristic_check(query, chunks)
        else:
            verdicts = self._llm_check(query, chunks, strictness)

        # Per-chunk composite score: relevance * sufficiency * consistency
        per_chunk_scores = [
            v.relevance * v.sufficiency * v.consistency for v in verdicts
        ]
        # Score using only the best N chunks (by composite score), not all retrieved chunks.
        # This stops irrelevant/filler chunks from dragging down a genuinely strong result.
        top_scores = sorted(per_chunk_scores, reverse=True)[:TOP_N_FOR_SCORING]
        overall_score = round(sum(top_scores) / len(top_scores), 4) if top_scores else 0.0

        threshold = PASS_THRESHOLDS[strictness]
        passed = overall_score >= threshold

        reason = (
            f"strictness={strictness}, retrieval_confidence={retrieval_confidence}, "
            f"overall_score={overall_score}, threshold={threshold}"
        )

        return VerificationResult(
            passed=passed,
            strictness=strictness,
            overall_score=overall_score,
            chunk_verdicts=verdicts,
            reason=reason,
        )


def main():
    """Quick manual test wiring retriever -> verifier together."""
    from retriever import Retriever

    r = Retriever()
    v = Verifier()

    query = input("Enter a test query: ")
    retrieval = r.retrieve_with_confidence(query, k=5)
    result = v.verify(query, retrieval)

    print(f"\nStrictness applied: {result.strictness}")
    print(f"Overall verification score: {result.overall_score}")
    print(f"PASSED: {result.passed}")
    print(f"Reason: {result.reason}\n")

    for vd in result.chunk_verdicts:
        print(
            f"  Chunk {vd.chunk_index}: relevance={vd.relevance} "
            f"sufficiency={vd.sufficiency} consistency={vd.consistency}"
        )
        if vd.reasoning:
            print(f"    reasoning: {vd.reasoning}")


if __name__ == "__main__":
    main()
