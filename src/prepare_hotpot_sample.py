"""
prepare_hotpot_sample.py
One-time data prep script: extracts a small sample from HotpotQA
(hotpot_dev_distractor_v1.json) and converts it into:

  1. data/*.txt          -- corpus passages for indexing.py to chunk & embed
  2. data/eval_set.json   -- questions + ground-truth answers + supporting facts,
                             used later by evaluate.py to measure accuracy/hallucination

Usage:
    python src/prepare_hotpot_sample.py
"""

import json
from pathlib import Path

# ---- Config ----
SOURCE_FILE = Path("hotpot_dev_distractor_v1.json")  # place this in the project root
DATA_DIR = Path("data")
NUM_QUESTIONS = 100


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE_FILE}. Place hotpot_dev_distractor_v1.json "
            f"in the project root (same folder as this script's parent)."
        )

    print(f"Loading {SOURCE_FILE}...")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    sample = full_data[:NUM_QUESTIONS]
    print(f"Using first {len(sample)} questions.")

    # Clear out old corpus files (keep the folder)
    DATA_DIR.mkdir(exist_ok=True)
    for old_file in DATA_DIR.glob("*.txt"):
        old_file.unlink()

    eval_set = []
    seen_titles = set()

    for item in sample:
        qid = item["_id"]
        question = item["question"]
        answer = item["answer"]
        qtype = item.get("type", "")
        level = item.get("level", "")
        supporting_facts = item["supporting_facts"]  # list of [title, sentence_index]

        # Write each unique context document as its own .txt file
        for title, sentences in item["context"]:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:80]
            file_path = DATA_DIR / f"{safe_name}.txt"
            text = " ".join(sentences)
            file_path.write_text(text, encoding="utf-8")

        eval_set.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "type": qtype,
            "level": level,
            "supporting_facts": supporting_facts,
        })

    eval_path = DATA_DIR / "eval_set.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"Wrote {len(seen_titles)} corpus .txt files to {DATA_DIR}/")
    print(f"Wrote {len(eval_set)} eval questions to {eval_path}")
    print("\nNext: run 'python src/indexing.py' to rebuild the index on this new corpus.")


if __name__ == "__main__":
    main()
