# DEVLOG — Domain Intelligence System
A combined **ML + RAG** system:
- ML predicts customer churn probability
- RAG answers questions using company telecom policies with citations

---

## Day 1 — Project Setup + Dataset Understanding (Churn Data)

###  Goal
Set the foundation like a real production repo:
- reproducible environment
- clean folder structure
- dataset inspection before touching ML

### Work done
#### 1) Repo + Environment Setup
- Initialized project repo: `domainIntelligenceSystem`
- Created virtual environment: `dis`
- Installed core dependencies:
  - pandas, numpy, scikit-learn, joblib
  - fastapi/uvicorn (for later API stage)

#### 2) Dataset Added
- Added churn dataset (`churn.csv`)
- Verified data loads correctly and inspected:
  - shape
  - column names
  - churn label distribution
  - missing values

#### 3) What I found in the dataset
- Target: `Churn Label` (`Yes/No`)
- Dataset size: `7043 rows × 50 columns`
- Class distribution:
  - No: 5174
  - Yes: 1869 (imbalanced dataset)
- Discovered key leakage columns:
  - `Churn Score`
  - `Churn Category`
  - `Churn Reason`
These were removed later because they directly or indirectly reveal churn outcome.

### Notes / Learnings
- Accuracy alone is useless in churn (imbalanced dataset).
- First priority is understanding data + leakage detection before modeling.

---

## Day 2 — Data Cleaning + Feature Selection + ML Baseline Models

### Goal
Train a trustworthy churn prediction model without data leakage.

### Work done
#### 1) Feature Cleaning (Column Dropping)
Dropped non-useful + leakage-heavy columns:
- ID:
  - `Customer ID`
- Leakage:
  - `Churn Score`, `Churn Category`, `Churn Reason`
- Location noise:
  - `City`, `Zip Code`, `Latitude`, `Longitude`

Saved cleaned dataset:
- `data/churn_clean.csv`
- Shape after cleaning: `7043 × 42`

#### 2) Missing Values Handling
Only two columns had missing values:
- `Offer` (3877 missing)
- `Internet Type` (1526 missing)

Instead of dropping rows, filled:
- `Offer` → `"None"`
- `Internet Type` → `"None"`

Reason:
Missing values here carry meaning (no offer / no internet service).

#### 3) Leakage Sanity Check (Very important)
Initial Logistic Regression gave extremely high score (~0.99 ROC AUC)
→ This felt suspicious, so I investigated further.

Found more risky columns:
- `Customer Status`
- `Satisfaction Score`
- `Total Revenue`
- `Total Refunds`
- `Quarter`

These were dropped because they likely contain post-event signals.

Final drop list:
- `Customer Status`
- `Quarter`
- `Satisfaction Score`
- `Total Revenue`
- `Total Refunds`

#### 4) ML Baseline Training
Trained 2 models:

 Logistic Regression (baseline + final candidate)
- ROC AUC: ~0.9096
- Accuracy: ~0.85
- Churn Recall: ~0.67

 Random Forest (non-linear model)
- ROC AUC: ~0.9057
- Accuracy: ~0.85
- Churn Recall: ~0.59

#### 5) Model Decision
Selected **Logistic Regression** because:
- higher churn recall (better at catching churn customers)
- explainable
- stable + lightweight for deployment

### Notes / Learnings
- Removing leakage reduced ROC AUC from unrealistic 0.99 to realistic 0.91.
- For churn prediction, **recall of churn class** is the main metric.

---

## Day 3 — Model Saving + Prediction Module + RAG Index (Document Intelligence)

### Goal
Make the project usable:
- save trained model
- predict churn anytime
- build RAG system on policy PDFs

### Work done
#### 1) Saving the Model
Saved final pipeline (preprocessing + model) using joblib:
- `models/churn_model.joblib`

This is important because:
- No need to re-run preprocessing during inference
- Same feature encoding is reused during prediction

#### 2) Prediction Script (Inference Ready)
Created:
- `ml/predict.py`

Features:
- loads saved model
- accepts a customer feature dictionary
- returns:
  - churn_prediction (0/1)
  - churn_probability
  - risk label (low/medium/high)

Sample output:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.9834,
  "risk": "high"
}
```
### 3) RAG Document Setup
Added telecom related policy PDFs:
- CustomerServicePolicy.pdf
- RefundPolicy.pdf
- Termination.pdf
- TermsCondition.pdf
- RetentionPolicy.pdf (customed created for controlled demo)

### 4) RAG Index Builder
Built rag/build_index.py which:
- reads PDFs from docs/
- extract text page-wise
- chunks text (800 chars, 150 overlap)
- generates embeddings(all-MiniLM-L6-V2)
- stores embeddings in FAISS index
- stores chunk metadata fro citations

Output:
- Found 5 PDFs
- Total chunks indexed: 455
- Index saved in : rag/index/docs.index rag/index/docs_meta.pkl

### 5) Retrieval Testing
Created rag/ask.py to:
- embed user query
- retrieve top-K relevant chunks from FAISS
- show results with citations (PDF+page)

Test query: What is the refund timeline?

Retrieved relevant chunks from:
- Termination.pdf (refund benchmark 60 days / 7 days)
- RefundPolicy.pdf (company refund description)

### Notes/Learnings
- Chunking + overlap is essential for accurate retrieval.
- FAISS is extremely fast and perfect for local vector search.
- Citations make RAG answers trustworthy and demo ready.

## Current Status (End of Day 3)
### Completed
- Churn ML pipeline(clean + leakage free)
- Saved model + inference module
- RAG indexing system with citation-based retrieval

### Next steps 
- Improve rag/ask.py to generate final answers(extractive/LLM)
- Combine ML + RAG (churn prediction + retention action recommendation from policy docs)
- FastAPI endpoints: /predict, /upload-docs, /ask


## Day 4 — ML + RAG Integration (Churn → Retention Recommendation)

### Goal
Combine both subsystems:
- ML predicts churn probability + risk level
- RAG retrieves retention actions from policy documents
- Output becomes business-usable: **risk + action plan + citations**

---

### Work Done

#### 1) Built integration module: `rag/recommend.py`
Created a new script that connects:
- `models/churn_model.joblib` (ML pipeline)
- FAISS vector index (`rag/index/docs.index`)
- metadata store (`rag/index/docs_meta.pkl`)

Flow:
1. Load churn model using joblib
2. Predict churn probability and classify risk (low/medium/high)
3. Create a RAG query based on risk
4. Retrieve relevant chunks from FAISS
5. Return policy-grounded actions + citations

---

#### 2) Improved RAG Retrieval Quality
Initially, retrieval returned definition sections (risk meaning).
So improved the retrieval query to include action keywords:
- `RET10`
- `discount`
- `plan upgrade`
- `premium support`
- `escalation timeline`

Also filtered results strictly to:
- `RetentionPolicy.pdf`
so the system consistently pulls retention actions rather than unrelated policy chunks.

---

#### 3) Added Action Chunk Selection Logic
Added keyword-based filtering over retrieved chunks to select the chunk most likely containing actual action steps.

Keywords used:
- `Offer`, `RET10`, `discount`, `Premium Support`, `Plan Upgrade`, `Escalation`, `within`

This improved output relevance significantly.

---

### Output Example
For a sample customer:
- churn_probability ≈ 0.9834
- risk = high

RAG successfully retrieved policy section:
- high-risk actions (RET10 discount, plan upgrade, premium support, contract lock-in)
and returned citations.

---

### Key Learnings
- Retrieval quality depends heavily on query wording
- Filtering docs by source improves reliability
- Combining ML + RAG is the core “enterprise feature”
  (prediction + explainable policy action plan)

---

### Next Steps
- Convert recommendation output into structured response:
  - bullet list actions
  - eligibility rules
  - clean citations
- Wrap everything in FastAPI:
  - `/predict`
  - `/ask`
  - `/recommend`
## Day 5 to Day 8 — FastAPI Backend + Structured Recommendation Output

### Goal
Convert ML + RAG scripts into a working backend API that can be tested using Swagger UI.

---

### Work Done

#### 1) Built FastAPI backend
Created `api/` package and implemented endpoints:
- `POST /predict` → churn prediction
- `POST /ask` → retrieve top-k document chunks from FAISS
- `POST /recommend` → churn prediction + retention policy recommendation with citations

Swagger UI available at:
- `/docs`

---

#### 2) Improved recommendation output (structured JSON)
Earlier `/recommend` returned raw policy chunks.
To make it production/demo-ready, created an action parser:

- Created: `rag/action_parser.py`
- Implemented: `parse_policy_actions()`

Now `/recommend` returns:
- `message` (human-friendly summary based on risk)
- `actions` list (structured format)
  - action title
  - details
  - eligibility
- `sources` (pdf + page citations)

Example fields:
```json
"actions": [
  {"title": "Offer RET10 Discount", "details": "...", "eligibility": "..."},
  {"title": "Plan Upgrade Offer", "details": "...", "eligibility": "..."}
]

