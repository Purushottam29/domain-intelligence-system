import joblib
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/churn_model.joblib"
INDEX_DIR = "rag/index"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_rag():
    index = faiss.read_index(f"{INDEX_DIR}/docs.index")
    with open(f"{INDEX_DIR}/docs_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, meta, embedder


def rag_search(query: str, top_k: int = 5):
    index, meta, embedder = load_rag()
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, ids = index.search(q_emb, top_k)

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        chunk = meta[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "distance": float(dist),
        })
    return results


def predict_churn(customer: dict):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([customer])

    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)

    risk = "low"
    if proba >= 0.7:
        risk = "high"
    elif proba >= 0.5:
        risk = "medium"

    return pred, proba, risk


def pick_action_chunk(results: list[dict]) -> dict | None:
    """
    From retrieved chunks, pick the one that most likely contains retention actions.
    """
    if not results:
        return None

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

    # Prefer chunks that contain action keywords
    for r in results:
        t = r["text"].lower()
        if any(k.lower() in t for k in keywords):
            return r

    # fallback: first result
    return results[0]


def format_actions(text: str, limit: int = 1600) -> str:
    """
    Converts raw policy chunk into readable bullet/action list.
    """
    if not text:
        return ""

    # Normalize whitespace
    text = " ".join(text.split())

    # Focus on section after "Recommended actions" if present
    marker = "Recommended actions:"
    if marker.lower() in text.lower():
        # case-insensitive split
        idx = text.lower().find(marker.lower())
        text = text[idx + len(marker):].strip()

    # Put newlines before numbered items
    for m in ["1.", "2.", "3.", "4.", "5.", "6."]:
        text = text.replace(m, f"\n{m}")

    # Add a newline before common bullets too (some PDFs extract like ○)
    text = text.replace("○", "\n  -")

    return text[:limit]


def main():
    print("=== Domain Intelligence System (ML + RAG) ===")

    # sample customer (edit later)
    customer = {
        "Gender": "Male",
        "Age": 35,
        "Under 30": "No",
        "Senior Citizen": "No",
        "Married": "Yes",
        "Dependents": "No",
        "Number of Dependents": 0,
        "Country": "United States",
        "State": "California",
        "Population": 500000,
        "Referred a Friend": "No",
        "Number of Referrals": 0,
        "Tenure in Months": 10,
        "Offer": "None",
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": 10.0,
        "Multiple Lines": "No",
        "Internet Service": "Yes",
        "Internet Type": "Fiber Optic",
        "Avg Monthly GB Download": 20,
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": "Month-to-Month",
        "Paperless Billing": "Yes",
        "Payment Method": "Bank Withdrawal",
        "Monthly Charge": 85.0,
        "Total Charges": 900.0,
        "Total Extra Data Charges": 0,
        "Total Long Distance Charges": 100.0,
        "CLTV": 3000.0
    }

    pred, proba, risk = predict_churn(customer)

    print("\n--- ML Prediction ---")
    print("Churn prediction:", pred)
    print("Churn probability:", round(proba, 4))
    print("Risk:", risk)

    # Better targeted RAG query (so it retrieves action section, not definitions)
    rag_query = (
        "High churn risk retention actions: RET10 discount, plan upgrade offer, premium support add-on, "
        "contract lock-in recommendation, escalation within 12 hours"
    )

    results = rag_search(rag_query, top_k=8)

    # filter only RetentionPolicy for this module
    results = [r for r in results if r["source"] == "RetentionPolicy.pdf"]

    if not results:
        print("\n❌ No relevant retention policy chunks found.")
        print("Tip: ensure RetentionPolicy.pdf is indexed and exists in docs/ folder.")
        return

    print("\n--- RAG Evidence (Top chunks) ---")
    for r in results[:3]:
        print(f"- {r['source']} (page {r['page']})")

    action_chunk = pick_action_chunk(results)

    print("\n--- Recommended Actions (Policy-grounded) ---")
    formatted = format_actions(action_chunk["text"])
    print(formatted)

    print("\nSources:")
    # Show unique citations only
    seen = set()
    for r in results[:5]:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            print(f"- {r['source']} (page {r['page']})")


if __name__ == "__main__":
    main()

