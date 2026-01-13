import pandas as pd

DROP_COLS = [
    "Customer Status",
    "Quarter",
    "Satisfaction Score",
    "Total Refunds",
    "Total Revenue",
]

TARGET_COL = "Churn Label"


def load_data(path: str = "data/churn_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # encode target
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})

    # drop leakage/risky cols
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df

