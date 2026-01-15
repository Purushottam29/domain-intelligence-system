#importing all the libraries
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = "rag/index"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_index():
    index = faiss.read_index(f"{INDEX_DIR}/docs.index")
    with open(f"{INDEX_DIR}/docs_meta.pkl","rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(query: str, top_k: int = 5):
    #load components
    index, meta = load_index()
    embedder = SentenceTransformer(MODEL_NAME)

    #embed query
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")

    #search faiss
    distances, ids = index.search(q_emb, top_k,)

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        chunk = meta[idx]
        results.append({
            "text":chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "distance": float(dist)
        })
    return results

def main():
    print("=== RAG Ask (Retrieval + Citations) ===")
    query = input("Ask a question: ").strip()

    if not query:
        print("Empty question. Exiting")
        return 

    results = search(query, top_k=5)

    print("\nTop retrieved chunks:\n")
    for i, r in enumerate(results,1):
        print(f"--- Result {i} ---")
        print(f"Source: {r['source']} | Page: {r['page']} | Distance: {r['distance']:.4f}")
        print(r["text"][:700])
        print()

    print("Retrieval done. Next step: connect an LLM to generate final answers.")


if __name__ == "__main__":
    main()
