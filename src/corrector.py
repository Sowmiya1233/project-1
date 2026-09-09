"""
corrector.py
Stage 5 of the pipeline: Self-Correction Loop

Responsibilities:
- When verification FAILS, reformulate the query using the LLM
  (broaden it, rephrase it, or break it into a more searchable form)
- Re-retrieve with the new query
- Re-verify the new evidence
- Repeat up to MAX_ITERATIONS times
- Log why each correction was triggered (useful for later ablation analysis)
- Stop early if verification passes, or give up gracefully after the cap

Usage:
    from retriever import Retriever
    from verifier import Verifier
    from corrector import Corrector

    r = Retriever()
    v = Verifier()
    c = Corrector(retriever=r, verifier=v)

    outcome = c.run(query="What is the boiling point of nitrogen?")
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"
MAX_ITERATIONS = 3  # cap to avoid infinite retrieval cycles


@dataclass
class CorrectionAttempt:
    iteration: int
    query_used: str
    retrieval_confidence: float
    verification_passed: bool
    verification_score: float
    strictness: str
    trigger_reason: str = ""  # why a correction was triggered after this attempt (empty if passed)


@dataclass
class CorrectionOutcome:
    final_passed: bool
    final_query: str
    final_retrieval: Dict
    final_verification: "object"  # VerificationResult from verifier.py
    attempts: List[CorrectionAttempt] = field(default_factory=list)
    gave_up: bool = False


class Corrector:
    def __init__(self, retriever, verifier, model_name: str = MODEL_NAME, max_iterations: int = MAX_ITERATIONS):
        self.retriever = retriever
        self.verifier = verifier
        self.client = Groq()  # reads GROQ_API_KEY from environment
        self.model_name = model_name
        self.max_iterations = max_iterations

    def _reformulate_query(self, original_query: str, current_query: str, chunk_summaries: List[str]) -> str:
        """
        Ask the LLM to rewrite the query so retrieval is more likely to succeed next time.
        Gives it context on what was retrieved last time so it can avoid repeating the same miss.
        """
        evidence_note = (
            "\n".join(f"- {s[:150]}" for s in chunk_summaries[:3])
            if chunk_summaries else "(no useful evidence retrieved)"
        )

        system_prompt = (
            "You are a query reformulation module in a self-corrective RAG pipeline. "
            "The previous retrieval attempt failed verification (evidence was insufficient, "
            "irrelevant, or inconsistent). Rewrite the query so it is more likely to retrieve "
            "relevant evidence from the corpus. You may broaden it, rephrase it, decompose it, "
            "or focus it on a different aspect. "
            "Respond ONLY with the rewritten query text, nothing else — no preamble, no quotes."
        )

        user_prompt = f"""Original user question: {original_query}
Most recent query tried: {current_query}

Evidence retrieved last time (top snippets):
{evidence_note}

Rewrite the query for the next retrieval attempt."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=150,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        new_query = response.choices[0].message.content.strip().strip('"')
        return new_query if new_query else current_query

    def run(self, query: str, k: int = 10) -> CorrectionOutcome:
        original_query = query
        current_query = query
        attempts: List[CorrectionAttempt] = []

        for iteration in range(1, self.max_iterations + 1):
            retrieval = self.retriever.retrieve_with_confidence(current_query, k=k)
            verification = self.verifier.verify(current_query, retrieval)

            passed = verification.passed

            if passed:
                attempts.append(CorrectionAttempt(
                    iteration=iteration,
                    query_used=current_query,
                    retrieval_confidence=retrieval["confidence"],
                    verification_passed=True,
                    verification_score=verification.overall_score,
                    strictness=verification.strictness,
                    trigger_reason="",
                ))
                return CorrectionOutcome(
                    final_passed=True,
                    final_query=current_query,
                    final_retrieval=retrieval,
                    final_verification=verification,
                    attempts=attempts,
                    gave_up=False,
                )

            # Verification failed -> log why a correction is being triggered
            trigger_reason = (
                f"verification failed at strictness={verification.strictness} "
                f"(score={verification.overall_score}); triggering re-retrieval"
            )
            attempts.append(CorrectionAttempt(
                iteration=iteration,
                query_used=current_query,
                retrieval_confidence=retrieval["confidence"],
                verification_passed=False,
                verification_score=verification.overall_score,
                strictness=verification.strictness,
                trigger_reason=trigger_reason,
            ))

            if iteration == self.max_iterations:
                # Out of attempts — give up gracefully, return the last (failed) result
                return CorrectionOutcome(
                    final_passed=False,
                    final_query=current_query,
                    final_retrieval=retrieval,
                    final_verification=verification,
                    attempts=attempts,
                    gave_up=True,
                )

            # Reformulate for the next attempt
            chunk_summaries = [c["text"] for c in retrieval["chunks"]]
            current_query = self._reformulate_query(original_query, current_query, chunk_summaries)

        # Should not reach here, but included for safety
        return CorrectionOutcome(
            final_passed=False,
            final_query=current_query,
            final_retrieval=retrieval,
            final_verification=verification,
            attempts=attempts,
            gave_up=True,
        )


def main():
    """Quick manual test wiring retriever -> verifier -> corrector together."""
    from retriever import Retriever
    from verifier import Verifier

    r = Retriever()
    v = Verifier()
    c = Corrector(retriever=r, verifier=v)

    query = input("Enter a test query: ")
    outcome = c.run(query)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {'PASSED' if outcome.final_passed else 'FAILED (gave up)' if outcome.gave_up else 'FAILED'}")
    print(f"Final query used: {outcome.final_query}")
    print(f"{'='*60}\n")

    print("Attempt log:")
    for a in outcome.attempts:
        status = "PASS" if a.verification_passed else "FAIL"
        print(f"  [Iter {a.iteration}] {status} | query=\"{a.query_used}\" | "
              f"retrieval_conf={a.retrieval_confidence} | verify_score={a.verification_score} "
              f"| strictness={a.strictness}")
        if a.trigger_reason:
            print(f"      -> {a.trigger_reason}")


if __name__ == "__main__":
    main()
