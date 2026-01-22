from typing import Dict, Any, List
import joblib
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from rag.action_parser import parse_policy_actions
from api.logger import get_logger

logger = get_logger()

# Paths
MODEL_PATH = "models/churn_model.joblib"
INDEX_DIR = "rag/index"
EMBED_MODEL = "all-MiniLM-L6-v2"

_model = joblib.load(MODEL_PATH)

_faiss_index = faiss.read_index(f"{INDEX_DIR}/docs.index")
with open(f"{INDEX_DIR}/docs_meta.pkl", "rb") as f:
    _docs_meta = pickle.load(f)

_embedder = SentenceTransformer(EMBED_MODEL)


def _risk_from_proba(proba: float) -> str:
    if proba >= 0.7:
        return "high"
    elif proba >= 0.5:
        return "medium"
    return "low"


def predict_service(customer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict churn for a single customer dict.
    customer keys must match training features.
    """
    X = pd.DataFrame([customer])

    proba = float(_model.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)
    risk = _risk_from_proba(proba)

    return {
        "churn_prediction": pred,
        "churn_probability": round(proba, 4),
        "risk": risk,
    }


def ask_service(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    RAG retrieval: returns top_k document chunks with metadata.
    """
    q_emb = _embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, ids = _faiss_index.search(q_emb, top_k)

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        chunk = _docs_meta[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": int(chunk["page"]),
            "distance": float(dist),
        })
    logger.info(f"RAG ASK query='{question}' | sources={[ (r['source'], r['page']) for r in results[:3] ]}")
    return results


def recommend_service(customer: Dict[str, Any]) -> Dict[str, Any]:
    """
    ML prediction + policy grounded recommendation text + citations.
    (No LLM: extractive + safe)
    """
    pred_out = predict_service(customer)
    risk = pred_out["risk"]

    # targeted query to retrieve "actions", not definitions
    if risk == "high":
        rag_query = (
                "High risk customers churn probability >= 0.70 retention actions: "
                "RET10 discount, plan upgrade offer, premium support add-on, contract lock-in, escalation within 12 hours"
        )
    elif risk == "medium":
        rag_query = (
                "Medium risk customers churn probability 0.50 to 0.69 retention actions: "
                "RET5 discount, service quality check, customer education, diagnostics"
                )
    else:
        rag_query = (
                "Low risk customers churn probability < 0.50 recommended actions: "
                "engagement newsletters loyalty benefits plan suggestions"
                )


    results = ask_service(rag_query, top_k=8)
    logger.info(f"RECOMMEND risk={risk} | sources={[ (r['source'], r['page']) for r in results[:3] ]}")

    # keep only retention policy chunks
    results = [r for r in results if r["source"] == "RetentionPolicy.pdf"]

    if not results:
        return {
            **pred_out,
            "recommended_text": "No retention policy evidence found in indexed documents.",
            "sources": []
        }

    # pick best chunk containing actions keywords
    keywords = [
        "Recommended actions",
        "RET10",
        "RET5",
        "Plan Upgrade",
        "Premium Support",
        "Contract Lock-in",
        "Escalation",
        "within 24 hours",
        "within 12 hours",
        "discount",
    ]

    action_chunk = results[0]
    for r in results:
        t = r["text"].lower()
        if any(k.lower() in t for k in keywords):
            action_chunk = r
            break

    # format recommendation
    text = " ".join(action_chunk["text"].split())
    actions = parse_policy_actions(action_chunk["text"])
    if risk == "low":
        actions = []
    marker = "Recommended actions:"
    if marker.lower() in text.lower():
        idx = text.lower().find(marker.lower())
        text = text[idx + len(marker):].strip()

    # nicer formatting
    for m in ["1.", "2.", "3.", "4.", "5.", "6."]:
        text = text.replace(m, f"\n{m}")
    text = text.replace("○", "\n  -")

    recommended_text = text[:2500]

    # sources (unique)
    sources = []
    seen = set()
    for r in results[:6]:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            sources.append(f"{r['source']} (page {r['page']})")
    
    if risk=="low":
        message = "Low churn risk. No discount offer required. Maintain engagement and loyalty benefits."
    elif risk == "medium":
        messgae = "Medium churn risk. Recommend light retentionactions like RET5 discount + service quality check."
    else:
        message = "High churn risk. Apply immediate retention actions (RET10, upgrade offers, premium support, escalation)"

    return {
        **pred_out,
        "message": message,
        "recommended_text": recommended_text,
        "sources": sources,
        "actions": actions
    }


