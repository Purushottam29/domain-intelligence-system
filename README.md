# Domain Intelligence System (ML + RAG)

A production-style **Domain Intelligence System** that combines:
- **Churn Prediction (ML)** — predicts whether a customer will churn + probability + risk level
- **Document Intelligence (RAG)** — answers questions from policy PDFs using FAISS retrieval with citations
- **Recommendation Engine (ML + RAG)** — predicts churn risk and suggests retention actions grounded in policy documents

This project simulates a real enterprise workflow:
> *Predict churn → justify actions using company policy docs → return citations.*

---

## Features

### 1) Churn Prediction (ML)
- Trained models:
  - Logistic Regression (final)
  - Random Forest (tested)
- Returns:
  - churn prediction (0/1)
  - churn probability
  - risk label (low/medium/high)

### 2) Document Intelligence (RAG Retrieval)
- Loads policy PDFs from `/docs`
- Extracts text → chunks → embeddings → FAISS index
- Answers questions with citations:
  - PDF name + page number

### 3) Recommendation (ML + RAG)
- Uses churn prediction result to decide recommendation strategy:
  - High risk → RET10, escalation, premium support, plan upgrades
  - Medium risk → RET5 + diagnostics + service quality checks
  - Low risk → engagement + loyalty benefits (no discount actions)
- Returns:
  - message
  - structured actions list
  - sources

### 4) FastAPI Backend (Swagger Ready)
Endpoints:
- `POST /predict`
- `POST /ask`
- `POST /recommend`

Swagger UI:
- `http://127.0.0.1:8000/docs`

### 5) Logging + Monitoring
Logs request data including:
- endpoint
- status code
- latency
- RAG query + top sources

Stored at:
- `logs/api.log`

### 6) API Key Authentication
All endpoints are protected using:
- Header: `X-API-Key`

---

## High-level Architecture

### A) ML Pipeline
`churn_clean.csv → preprocessing → model training → saved pipeline (.joblib)`

### B) RAG Pipeline
`PDFs → text extraction → chunking → embeddings → FAISS index`

### C) Combined Workflow
`Customer JSON → churn risk → policy search → retention actions + citations`

![Architecture](https://github.com/Purushottam29/domain-intelligence-system/blob/cd7c6b9e8b002780094546dcb272793c9874f15f/assets/Architecture_Diagram_Text.png)
---

## Folder Structure

## Setup (Local)

### 1) Create Virtual Environment
```bash
python3 - m venv dis
source dis/bin/activate
```

### 2) Install Dependencies
```bash
pip install -r requirements.txt
```

### 3) ML Training

#### Train Logistic Regression
```bash
python ml/baseline_logreg.py
```
Note: Model will be saved inside model directory along with the scores
![Ouput1](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/logistic_regression_output.png)
Saved model:
- models/churn_model.joblib

#### Train Random Forest 
```bash
python ml/random_forest.py
```
Note: No model will be saved only the scores will be shown
```
![Output2](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/Random_Forest_Output.png)

### 4) Build RAG Index (FAISS)
Place PDFs inside docs/ then run
```bash
python rag/build_index.py
```
This generates:
- rag/index/docs.index
- rag/index/docss_meta.pkl

### 5) Test retrieval (RAG)
Run:
```bash
python rag/ask.py
```
Example questions:
- What is the refund timeline?
- What is the termination process?
- What retention actions apply to high risk customers?

### 6) Run FastAPI Backend
Start API with API Key
```bash 
API_KEY = "puru123" uvicorn api.main:app --reload
```
Open Swagger:
- http://127.0.0.1:8000/docs

## Authentication
All API request must include header:
```bash
X-API-Key: puru123
```
Without this key:
- returns 401 Unauthorized
![API_Key_Authentication](https://github.com/Purushottam29/domain-intelligence-system/blob/a619cd0cfb2ce05dc74c4940b9faea452ee51784/assets/API_authenticate.png)

## API Endpoints

### 1) POST/predict
Predict churn for a customer.
Input: customer JSON
Output: churn prediction + probability + risk

Sample Input:
```bash
{
  "Gender": "Female",
  "Age": 50,
  "Under 30": "No",
  "Senior Citizen": "No",
  "Married": "Yes",
  "Dependents": "Yes",
  "Number of Dependents": 2,
  "Country": "United States",
  "State": "Texas",
  "Population": 1200000,
  "Referred a Friend": "Yes",
  "Number of Referrals": 3,
  "Tenure in Months": 72,
  "Offer": "Offer B",
  "Phone Service": "Yes",
  "Avg Monthly Long Distance Charges": 2.5,
  "Multiple Lines": "Yes",
  "Internet Service": "Yes",
  "Internet Type": "DSL",
  "Avg Monthly GB Download": 10,
  "Online Security": "Yes",
  "Online Backup": "Yes",
  "Device Protection Plan": "Yes",
  "Premium Tech Support": "Yes",
  "Streaming TV": "No",
  "Streaming Movies": "No",
  "Streaming Music": "No",
  "Unlimited Data": "No",
  "Contract": "Two Year",
  "Paperless Billing": "No",
  "Payment Method": "Credit Card",
  "Monthly Charge": 40.0,
  "Total Charges": 2200.0,
  "Total Extra Data Charges": 0,
  "Total Long Distance Charges": 30.0,
  "CLTV": 6000.0
}
```
![Predict Input](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/Predict_Input.png)
![Predict Output](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/Predict_Input.png)

### 2) POST/ask
Ask a question from the policy documents.
Input:
```bash
{ "question": "What is the refund timeline?", "top_k": 5 }
```
Output: top chunks + citations
![Ask Input](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/Ask_Input.png)
![Ask Output](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/ask_output.png)


### 3) POST/recommend
Churn risk -> retention recommendation grounded in policy docs.
Input : Customer JSON (Take same which was given for predict)
Output Includes:
- churn probability
- risk
- message
- action list(structured)
- sources
![Recommend Input](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/Recommend_Input.png)
![Recommend Output](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/Recommend_Output.png)
![Recommend Output Terminal](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/Recommend_terminal.png)
## Logs
Logs stored at:
```bash
logs/api.log
```
View logs:
```bash
tail -n 50 logs/api.log
```
![Logs](https://github.com/Purushottam29/domain-intelligence-system/blob/bafd8aca88736d9426deff870c6aa27b7b095dee/assets/logs.png)

## Current Status:
- ML churn prediction pipeline
- Clean preprocessing + leakage handling
- FAISS RAG indexing with citations
- ML + RAG recommendation module
- FastAPI endpoints working
- Structured recommendations
- Logging + monitoring
- API key authentication

## Future Improvements
- Add LLM answer generation
- Add automated evaluation for RAG retrieval quality
- Deployment
- Add frontend dashboard for upload+query
- Improve action parser to support multi policy action reasoning

## Author
Purushottam Choudhar## Author
### Purushottam Choudhary
B.Tech Computer Science
* Github: https://github.com/Purushottam29
* LinkedIN: https://www.linkedin.com/in/purushottam-choudhary-166120373
* Mail: purushottamchoudhary2910@gmail.com
y 
