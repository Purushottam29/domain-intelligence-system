import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.joblib"

model = joblib.load(MODEL_PATH) #I did this so that model loads only once else every time predict is called model loads

def predict_one(customer: dict) -> dict:
    # model expects a dataframe
    X = pd.DataFrame([customer])

    proba = model.predict_proba(X)[0][1]   # probability of churn = 1
    pred = int(proba >= 0.5)

    risk = "low"
    if proba >= 0.7:
        risk = "high"
    elif proba >= 0.5:
        risk = "medium"

    return {
        "churn_prediction": pred,
        "churn_probability": round(float(proba), 4),
        "risk": risk
    }


if __name__ == "__main__":
    # sample customer
    sample_customer = {
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

    result = predict_one(sample_customer)
    print(result)

