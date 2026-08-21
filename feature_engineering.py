"""
Feature engineering used identically at training time and inference (Gradio) time.
Keeping this logic in one shared module guarantees train/serve consistency.
"""

import numpy as np
import pandas as pd

TARGET_COLUMN = "Is_Fraudulent"

DATE_COLUMNS = ["Claim_Date", "Service_Date", "Policy_Expiration_Date"]

# Columns that are identifiers / leakage-prone / raw text and should never
# be fed into the model directly.
DROP_COLUMNS = [
    "Patient_ID",
    "Policy_Number",
    "Claim_ID",
    "Hospital_ID",
    "Claim_Date",
    "Service_Date",
    "Policy_Expiration_Date",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Take the raw claim schema and return a dataframe of model-ready
    features. Safe to call on a single-row dataframe (inference) or the
    full training dataframe.
    """
    df = df.copy()

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Time-based features -------------------------------------------------
    df["Days_Service_To_Claim"] = (df["Claim_Date"] - df["Service_Date"]).dt.days
    df["Days_To_Policy_Expiration"] = (
        df["Policy_Expiration_Date"] - df["Claim_Date"]
    ).dt.days
    df["Claim_Year"] = df["Claim_Date"].dt.year
    df["Claim_Month"] = df["Claim_Date"].dt.month
    df["Claim_DayOfWeek"] = df["Claim_Date"].dt.dayofweek
    df["Claim_Is_Weekend"] = df["Claim_DayOfWeek"].isin([5, 6]).astype(int)
    df["Policy_Already_Expired"] = (df["Days_To_Policy_Expiration"] < 0).astype(int)

    # --- Ratio / derived numeric features ------------------------------------
    df["Claim_Amount_Per_Day"] = df["Claim_Amount"] / df["Length_of_Stay_Days"].replace(
        0, 1
    )
    df["Claim_Amount_Per_Procedure"] = df["Claim_Amount"] / df[
        "Number_of_Procedures"
    ].replace(0, 1)
    df["Total_OutOfPocket"] = df["Deductible_Amount"] + df["CoPay_Amount"]
    df["OutOfPocket_Ratio"] = df["Total_OutOfPocket"] / df["Claim_Amount"].replace(0, 1)
    df["Same_State"] = (df["Patient_State"] == df["Provider_State"]).astype(int)
    df["Same_City"] = (df["Patient_City"] == df["Provider_City"]).astype(int)
    df["High_Previous_Claims_Provider"] = (
        (
            df["Number_of_Previous_Claims_Provider"]
            > df["Number_of_Previous_Claims_Provider"].median()
        ).astype(int)
        if df["Number_of_Previous_Claims_Provider"].notna().any()
        else 0
    )
    df["Claim_Submitted_Late"] = df["Claim_Submitted_Late"].astype(int)

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    # Fill any residual NaNs created by date math (e.g. bad/missing dates)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df
