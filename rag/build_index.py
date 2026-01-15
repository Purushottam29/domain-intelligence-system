import os
from pathlib import Path
import pickle

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

DOCS_DIR = "docs"
INDEX_DIR = "rag/index"
MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_text_from_pdf(pdf_path: str)-> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_num": i+1, "text":text})
    return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start<0:
            start = 0;
    return chunks

def main():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    documents = []
    pdf_files = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])

    if not pdf_files:
        raise RuntimeError("No PDFs found in docs/ folder.")
    print("Found PDFs:", pdf_files)

    for pdf in pdf_files:
        pdf_path = os.path.join(DOCS_DIR, pdf)
        pages = extract_text_from_pdf(pdf_path)

        for page in pages:
            chunks = chunk_text(page["text"])
            for chunk in chunks:
                documents.append({
                    "text": chunk,
                    "source": pdf,
                    "page": page["page_num"]
                    })
    print("Total chunks:", len(documents))

    embedder = SentenceTransformer(MODEL_NAME)
    texts = [d["text"] for d in documents]

    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.index"))

    with open(os.path.join(INDEX_DIR, "docs_meta.pkl"), "wb") as f:
        pickle.dump(documents, f)

    print("Index saved to rag/index/")

if __name__=="__main__":
    main()
