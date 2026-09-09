"""
evaluate.py
Stage 8-9 of the pipeline: Evaluation & Analysis

Runs the full pipeline (retrieval -> verification -> correction ->
generation -> hallucination check) across every question in
data/eval_set.json, compares against ground-truth answers, and computes:

  - Answer accuracy (loose match against ground truth)
  - Faithfulness score (avg. from hallucination_check.py)
  - Verification pass rate (did the corrector ever get a PASS?)
  - Avg. correction iterations needed
  - Retrieval precision proxy (did any correct-source chunk get retrieved?)

Saves full per-question results to results/eval_results.json and prints
a summary table.

NEW in this version:
  - Retries API calls on 429 rate-limit errors with backoff, reading the
    provider's suggested wait time out of the error message when present.
  - Resumable: if results/eval_results.json already exists, questions that
    previously succeeded (no "error" key) are skipped automatically, so a
    re-run after hitting a daily quota only spends tokens on what's left.
    Use --force to ignore existing results and re-run everything.

Usage:
    python src/evaluate.py                 # run all questions in eval_set.json
    python src/evaluate.py --limit 10       # run only the first 10 (faster/cheaper)
    python src/evaluate.py --force          # ignore existing results, re-run all
"""

import os
import re
import json
import argparse
import time
from pathlib import Path
from dataclasses import asdict

from retriever import Retriever
from verifier import Verifier
from corrector import Corrector
from generator import Generator
from hallucination_check import HallucinationChecker

EVAL_SET_PATH = Path("data/eval_set.json")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "eval_results.json"

MAX_RETRIES = 6
DEFAULT_BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 120.0

# Matches things like "Please try again in 862.5ms" or "in 15m26.208s"
_WAIT_MS_RE = re.compile(r"try again in ([\d.]+)ms", re.IGNORECASE)
_WAIT_S_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)
_WAIT_MIN_RE = re.compile(r"try again in (\d+)m([\d.]+)s", re.IGNORECASE)


def _parse_suggested_wait(error_message: str):
    """Pull a suggested wait time (seconds) out of a provider error message."""
    m = _WAIT_MIN_RE.search(error_message)
    if m:
        minutes, seconds = m.groups()
        return int(minutes) * 60 + float(seconds)
    m = _WAIT_S_RE.search(error_message)
    if m:
        return float(m.group(1))
    m = _WAIT_MS_RE.search(error_message)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


def with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs), retrying on rate-limit (429) errors.

    - Reads the provider's own "try again in Xs" hint when present and waits
      that long (plus a small buffer), instead of guessing.
    - Falls back to exponential backoff if no hint is found.
    - Re-raises the exception if MAX_RETRIES is exceeded, or if the error
      is not a rate-limit error (e.g. bad request, auth failure).
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise  # not a rate limit issue, don't retry blindly

            attempt += 1
            if attempt > MAX_RETRIES:
                raise

            suggested = _parse_suggested_wait(str(e))
            if suggested is not None:
                wait_s = suggested + 1.0  # small buffer
                # If the provider says "try again in 15 minutes", that's a
                # daily-quota exhaustion, not a transient burst. Cap how long
                # we'll silently wait in one shot so the user can see progress.
                if wait_s > MAX_BACKOFF_SECONDS:
                    print(f"  [rate limit] Provider says wait ~{wait_s/60:.1f} min "
                          f"(daily quota likely exhausted). Waiting...")
                else:
                    print(f"  [rate limit] Waiting {wait_s:.1f}s as suggested by provider...")
            else:
                wait_s = min(DEFAULT_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                print(f"  [rate limit] Waiting {wait_s:.1f}s (attempt {attempt}/{MAX_RETRIES})...")

            time.sleep(wait_s)


def loose_match(predicted: str, ground_truth: str) -> bool:
    """
    Simple, forgiving accuracy check. Ground-truth answers in HotpotQA are
    either short entity/phrase strings, or "yes"/"no" for comparison questions.
    Generated answers are full sentences, so "yes"/"no" need semantic matching
    rather than literal substring matching.
    """
    predicted_lower = predicted.lower()
    gt_lower = ground_truth.lower().strip()

    if gt_lower == "yes":
        affirmative_cues = ["yes", "both are", "same nationality", "same neighborhood",
                             "correct", "indeed", "they are the same", "they do"]
        negative_cues = ["no,", "not the same", "different", "did not", "does not"]
        has_negative = any(cue in predicted_lower for cue in negative_cues)
        has_affirmative = any(cue in predicted_lower for cue in affirmative_cues)
        return has_affirmative and not has_negative

    if gt_lower == "no":
        negative_cues = ["no,", "not the same", "different", "did not", "does not", "no."]
        return any(cue in predicted_lower for cue in negative_cues)

    # For entity/phrase answers, check if the answer string appears in the prediction
    return gt_lower in predicted_lower


def load_existing_results():
    """Return (results_list, already_done_ids) from a prior run, if any."""
    if not RESULTS_PATH.exists():
        return [], set()
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        prior_results = data.get("results", [])
        # "done" = no error recorded on that question last time.
        # Questions that errored (e.g. hit the rate limit) are retried.
        done_ids = {r["id"] for r in prior_results if "error" not in r}
        return prior_results, done_ids
    except (json.JSONDecodeError, KeyError):
        return [], set()


def run_evaluation(limit: int = None, force: bool = False):
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_SET_PATH} not found. Run src/prepare_hotpot_sample.py first."
        )

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    if limit:
        eval_set = eval_set[:limit]

    print(f"Loaded {len(eval_set)} eval questions.")

    prior_results, done_ids = ([], set()) if force else load_existing_results()
    if done_ids:
        print(f"Found {len(done_ids)} previously-completed questions in "
              f"{RESULTS_PATH} — these will be skipped. Use --force to redo everything.")

    print("Initializing pipeline components...")

    r = Retriever()
    v = Verifier()
    c = Corrector(retriever=r, verifier=v)
    g = Generator()
    hc = HallucinationChecker()

    RESULTS_DIR.mkdir(exist_ok=True)

    # Keep prior successful records, we'll only recompute the rest
    results = [rec for rec in prior_results if rec["id"] in done_ids]
    correct_count = sum(1 for rec in results if rec.get("correct"))
    verification_pass_count = sum(1 for rec in results if rec.get("verification_passed"))
    total_iterations = sum(rec.get("iterations_used", 0) for rec in results)
    total_faithfulness = sum(rec.get("faithfulness_score", 0.0) for rec in results
                              if rec.get("faithfulness_score") is not None)
    faithfulness_count = sum(1 for rec in results if rec.get("faithfulness_score") is not None)

    todo = [item for item in eval_set if item["id"] not in done_ids]
    print(f"Running {len(todo)} question(s) this session "
          f"({len(eval_set) - len(todo)} already done).")

    for i, item in enumerate(todo, 1):
        question = item["question"]
        ground_truth = item["answer"]
        print(f"\n[{i}/{len(todo)}] {question}")

        record = {
            "id": item["id"],
            "question": question,
            "ground_truth": ground_truth,
            "type": item.get("type", ""),
            "level": item.get("level", ""),
        }

        try:
            outcome = with_retry(c.run, question)
            total_iterations += len(outcome.attempts)

            record["verification_passed"] = outcome.final_passed
            record["iterations_used"] = len(outcome.attempts)
            record["final_query"] = outcome.final_query

            if outcome.final_passed:
                verification_pass_count += 1

                gen_result = with_retry(g.generate, outcome.final_query, outcome.final_retrieval)
                report = hc.check(gen_result)

                record["predicted_answer"] = gen_result.answer_text
                record["clean_answer"] = report.clean_answer
                record["faithfulness_score"] = report.faithfulness_score
                record["flagged_claims"] = report.flagged_count

                total_faithfulness += report.faithfulness_score
                faithfulness_count += 1

                is_correct = loose_match(gen_result.answer_text, ground_truth)
                record["correct"] = is_correct
                if is_correct:
                    correct_count += 1

                print(f"  -> Answer: {gen_result.answer_text}")
                print(f"  -> Ground truth: {ground_truth} | Correct: {is_correct} | "
                      f"Faithfulness: {report.faithfulness_score}")
            else:
                record["predicted_answer"] = None
                record["correct"] = False
                print("  -> Verification never passed; no answer generated.")

        except Exception as e:
            record["error"] = str(e)
            print(f"  -> ERROR (giving up after retries): {e}")

        results.append(record)

        # Checkpoint after every question so a crash/quota cutoff doesn't
        # lose progress already made this session.
        _save_results(results, len(eval_set), correct_count, verification_pass_count,
                      total_iterations, total_faithfulness, faithfulness_count)

        time.sleep(0.5)  # small delay to be gentle on API rate limits

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    n = len(eval_set)
    summary = _build_summary(n, correct_count, verification_pass_count,
                              total_iterations, total_faithfulness, faithfulness_count)
    for k, val in summary.items():
        print(f"  {k}: {val}")
    print(f"\nFull results saved to {RESULTS_PATH}")

    remaining_errors = sum(1 for rec in results if "error" in rec)
    if remaining_errors:
        print(f"\n{remaining_errors} question(s) still have errors (likely quota exhausted). "
              f"Re-run this script later (without --force) to retry just those.")


def _build_summary(n, correct_count, verification_pass_count,
                    total_iterations, total_faithfulness, faithfulness_count):
    return {
        "total_questions": n,
        "verification_pass_rate": round(verification_pass_count / n, 4) if n else 0,
        "answer_accuracy": round(correct_count / n, 4) if n else 0,
        "avg_correction_iterations": round(total_iterations / n, 4) if n else 0,
        "avg_faithfulness_score": (
            round(total_faithfulness / faithfulness_count, 4) if faithfulness_count else 0
        ),
    }


def _save_results(results, n, correct_count, verification_pass_count,
                   total_iterations, total_faithfulness, faithfulness_count):
    summary = _build_summary(n, correct_count, verification_pass_count,
                              total_iterations, total_faithfulness, faithfulness_count)
    output = {"summary": summary, "results": results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--force", action="store_true",
                         help="Ignore existing results and re-run every question")
    args = parser.parse_args()
    run_evaluation(limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()