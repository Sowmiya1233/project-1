"""
retriever.py
Stage 3 of the pipeline: Initial Retrieval

Responsibilities:
- Load the persistent Chroma index built by indexing.py
- Embed an incoming query
- Return the top-k most relevant chunks (with scores + metadata)

Usage:
    from retriever import Retriever
    r = Retriever()
    results = r.retrieve("What causes hallucination in RAG systems?", k=5)
"""

from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb

# ---- Config (keep in sync with indexing.py) ----
# Anchored to this file's location (not the current working directory), so
# it resolves correctly regardless of where the process is launched from
# (local `streamlit run src/app.py` vs. a container's working directory).
INDEX_DIR = Path(__file__).resolve().parent.parent / "chroma_index"
COLLECTION_NAME = "corpus_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


class Retriever:
    def __init__(self, persist_dir: Path = INDEX_DIR, collection_name: str = COLLECTION_NAME):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            self.collection = client.get_collection(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not load collection '{collection_name}' from {persist_dir}/. "
                f"Did you run indexing.py first?"
            ) from e

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Return top-k chunks for a query, each with text, metadata, and a similarity score."""
        query_embedding = self.model.encode([query], convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for text, meta, dist in zip(docs, metas, dists):
            # Chroma returns distance (lower = more similar); convert to a 0-1 similarity score
            similarity = 1 / (1 + dist)
            chunks.append({
                "text": text,
                "metadata": meta,
                "distance": dist,
                "similarity": round(similarity, 4),
            })

        # Highest similarity first
        chunks.sort(key=lambda c: c["similarity"], reverse=True)
        return chunks

    def retrieve_with_confidence(self, query: str, k: int = DEFAULT_TOP_K) -> Dict:
        """
        Retrieve chunks plus a simple retrieval-confidence signal, useful later for
        the adaptive verification module (stage 4) to decide how strict to be.
        """
        chunks = self.retrieve(query, k)
        if not chunks:
            return {"chunks": [], "confidence": 0.0}

        top_score = chunks[0]["similarity"]
        avg_score = sum(c["similarity"] for c in chunks) / len(chunks)
        # crude confidence heuristic: blend of top-hit strength and overall agreement
        confidence = round(0.6 * top_score + 0.4 * avg_score, 4)

        return {"chunks": chunks, "confidence": confidence}


def main():
    """Quick manual test from the command line."""
    r = Retriever()
    query = input("Enter a test query: ")
    result = r.retrieve_with_confidence(query, k=5)

    print(f"\nRetrieval confidence: {result['confidence']}\n")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"[{i}] similarity={chunk['similarity']}  source={chunk['metadata']['source']}")
        print(f"    {chunk['text'][:150]}...\n")


if __name__ == "__main__":
    main()