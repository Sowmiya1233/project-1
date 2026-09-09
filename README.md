# Self-Corrective RAG with Adaptive Evidence Verification

A retrieval-augmented generation pipeline that verifies retrieved evidence
before generating an answer, and self-corrects (re-retrieves) when evidence
is insufficient or conflicting — aimed at reducing hallucination.

## Project structure

```
self-corrective-rag/
├── data/              # your corpus (.txt / .md files) — sample.txt included
├── src/
│   ├── indexing.py    # chunk corpus + build vector index
│   └── retriever.py   # query the index, get top-k chunks + confidence
├── notebooks/         # scratch space for experiments
├── results/           # evaluation outputs
├── requirements.txt
├── .env.example        # copy to .env and add your API key
└── .gitignore
```

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your Anthropic API key:
   ```
   cp .env.example .env
   ```

## Running the pipeline so far

1. Add your own corpus files to `data/` (a sample file is already included
   so you can test immediately).

2. Build the index:
   ```
   python src/indexing.py
   ```

3. Test retrieval:
   ```
   python src/retriever.py
   ```
   You'll be prompted for a query, and it will print the top-k matching
   chunks with similarity scores and a retrieval-confidence estimate.

## Pipeline stages (full plan)

- [x] 1. Corpus preparation & indexing (`indexing.py`)
- [x] 2. Initial retrieval (`retriever.py`)
- [ ] 3. Adaptive evidence verification (`verifier.py`) — scores relevance,
      sufficiency, consistency; adapts strictness to retrieval confidence
- [ ] 4. Self-correction loop (`corrector.py`) — re-retrieval on failed
      verification, with a max iteration cap
- [ ] 5. Answer generation (`generator.py`) — generates with per-claim
      citations to retrieved passages
- [ ] 6. Post-hoc hallucination check — decomposes answer into claims,
      checks each against cited evidence
- [ ] 7. Evaluation (`evaluate.py`) — EM/F1, faithfulness rate, citation
      precision/recall, retrieval precision@k, vs. vanilla RAG baseline

## Next steps

Ask your assistant to build `verifier.py` next to continue the pipeline.
