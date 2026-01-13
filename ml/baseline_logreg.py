import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

from utils import load_data, TARGET_COL


def main():
    df = load_data("data/churn_clean.csv")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, "models/churn_model.joblib")
    print("\nSaved model to models/churn_model.joblib")

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    print("\nMODEL: Logistic Regression")
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("\nClassification report:\n", classification_report(y_test, pred))


if __name__ == "__main__":
    main()

