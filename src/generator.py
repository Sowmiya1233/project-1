"""
generator.py
Stage 6 of the pipeline: Answer Generation

Responsibilities:
- Take evidence that has PASSED verification (from corrector.py's outcome)
- Generate an answer using ONLY that verified evidence
- Force structured output: each claim in the answer must cite which
  chunk(s) support it -- this is what makes later hallucination-checking
  (stage 7) possible, since every claim already points at its evidence.

Usage:
    from retriever import Retriever
    from verifier import Verifier
    from corrector import Corrector
    from generator import Generator

    r = Retriever()
    v = Verifier()
    c = Corrector(retriever=r, verifier=v)
    g = Generator()

    outcome = c.run("Were Scott Derrickson and Ed Wood of the same nationality?")
    if outcome.final_passed:
        result = g.generate(outcome.final_query, outcome.final_retrieval)
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


@dataclass
class Claim:
    text: str
    supporting_chunk_indices: List[int] = field(default_factory=list)


@dataclass
class GenerationResult:
    answer_text: str            # the full answer, human-readable
    claims: List[Claim]         # answer broken into individual claims with citations
    used_chunks: Dict[int, Dict]  # chunk_index -> chunk data, for citation lookup/display
    raw_model_output: str = ""


class Generator:
    def __init__(self, model_name: str = MODEL_NAME):
        self.client = Groq()  # reads GROQ_API_KEY from environment
        self.model_name = model_name

    def generate(self, query: str, retrieval: Dict) -> GenerationResult:
        """
        `retrieval` is the dict from Retriever.retrieve_with_confidence():
        {"chunks": [...], "confidence": float}
        Only call this after verification has PASSED for this retrieval.
        """
        chunks = retrieval["chunks"]
        chunk_block = "\n\n".join(
            f"[Chunk {i}] {c['text']}" for i, c in enumerate(chunks)
        )

        system_prompt = (
            "You are an answer generation module in a hallucination-resistant RAG "
            "pipeline. You must answer using ONLY the provided evidence chunks -- "
            "never use outside knowledge, never guess, never fill gaps with assumptions. "
            "If the evidence does not fully support a claim, do not make that claim. "
            "Every claim in your answer must be traceable to specific chunk(s). "
            "Respond ONLY with valid JSON, no preamble, no markdown fences."
        )

        user_prompt = f"""Question: {query}

Evidence chunks:
{chunk_block}

Write a concise, direct answer to the question using only this evidence.
Break your answer into individual claims. For each claim, list which chunk
indices support it.

Return JSON in exactly this shape:
{{
  "answer": "the full answer as a natural sentence or two",
  "claims": [
    {{"text": "a single claim from the answer", "supporting_chunks": [0, 2]}},
    ...
  ]
}}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content.strip()
        raw_output = text
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to salvage the "answer" field at minimum, even if claims array is broken
            match = re.search(r'"answer":\s*"([^"]*)"', text)
            fallback_answer = match.group(1) if match else (
                "Could not generate a structured answer (parse error)."
            )
            return GenerationResult(
                answer_text=fallback_answer,
                claims=[],
                used_chunks={i: c for i, c in enumerate(chunks)},
                raw_model_output=raw_output,
            )

        answer_text = parsed.get("answer", "")
        claims = [
            Claim(text=c.get("text", ""), supporting_chunk_indices=c.get("supporting_chunks", []))
            for c in parsed.get("claims", [])
        ]

        return GenerationResult(
            answer_text=answer_text,
            claims=claims,
            used_chunks={i: c for i, c in enumerate(chunks)},
            raw_model_output=raw_output,
        )


def format_result(result: GenerationResult) -> str:
    """Pretty-print an answer with its citations, showing the actual cited text."""
    lines = [f"ANSWER: {result.answer_text}\n", "Claims & citations:"]
    for claim in result.claims:
        sources = ", ".join(
            f"[{i}] {result.used_chunks[i]['metadata']['source']}"
            for i in claim.supporting_chunk_indices
            if i in result.used_chunks
        )
        lines.append(f"  - \"{claim.text}\"")
        lines.append(f"      supported by: {sources or '(no citation given)'}")
    return "\n".join(lines)


def main():
    """Quick manual test wiring the full pipeline together."""
    from retriever import Retriever
    from verifier import Verifier
    from corrector import Corrector

    r = Retriever()
    v = Verifier()
    c = Corrector(retriever=r, verifier=v)
    g = Generator()

    query = input("Enter a test query: ")
    outcome = c.run(query)

    if not outcome.final_passed:
        print("\nVerification never passed after correction attempts -- refusing to generate "
              "an answer rather than risk hallucination.")
        return

    print(f"\nUsing query: \"{outcome.final_query}\" (verification passed)\n")
    result = g.generate(outcome.final_query, outcome.final_retrieval)
    print(format_result(result))


if __name__ == "__main__":
    main()
