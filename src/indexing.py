"""
indexing.py
Stage 2 of the pipeline: Corpus Preparation & Indexing

Responsibilities:
- Load raw documents from data/
- Chunk them into overlapping passages
- Embed each chunk
- Build a persistent vector index (Chroma) for retrieval

Run directly to (re)build the index:
    python src/indexing.py
"""

import os
import json
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ---- Config ----
DATA_DIR = Path("data")
INDEX_DIR = Path("chroma_index")
COLLECTION_NAME = "corpus_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # fast, solid baseline; swap for a stronger model later
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between consecutive chunks


def load_documents(data_dir: Path) -> List[Dict]:
    """Load .txt/.md files from data_dir. Each file becomes one document."""
    docs = []
    for path in data_dir.glob("**/*"):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"doc_id": path.stem, "source": str(path), "text": text})
    if not docs:
        raise FileNotFoundError(
            f"No .txt or .md files found in {data_dir}. Add your corpus files there first."
        )
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple fixed-size sliding-window chunking with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # step forward, keeping overlap
    return chunks


def build_chunk_records(docs: List[Dict]) -> List[Dict]:
    """Turn documents into chunk-level records with stable IDs."""
    records = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{doc['doc_id']}_chunk{i}",
                "text": chunk,
                "metadata": {"doc_id": doc["doc_id"], "source": doc["source"], "chunk_index": i},
            })
    return records


def build_index(records: List[Dict], persist_dir: Path = INDEX_DIR):
    """Embed chunk records and store them in a persistent Chroma collection."""
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    client = chromadb.PersistentClient(path=str(persist_dir))
    # Fresh collection each build; drop if it already exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[r["id"] for r in records],
        documents=texts,
        metadatas=[r["metadata"] for r in records],
        embeddings=embeddings,
    )
    print(f"Indexed {len(records)} chunks into '{COLLECTION_NAME}' at {persist_dir}/")
    return collection


def main():
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents from {DATA_DIR}/")
    records = build_chunk_records(docs)
    print(f"Created {len(records)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    build_index(records)


if __name__ == "__main__":
    main()
