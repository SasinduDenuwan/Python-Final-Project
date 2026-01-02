import pandas as pd
import numpy as np
import os

# ---------------- PATH CONFIGURATION ---------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "diabetic_data.csv")
ID_MAP_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "IDs_mapping.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "cleaned_data.csv")


def load_data():
    """Load raw diabetes dataset with standardized NaNs."""
    df = pd.read_csv(RAW_DATA_PATH)
    df.replace("?", np.nan, inplace=True)
    return df


def drop_high_missing_columns(df):
    """Drop columns with excessive missing values (>90%)."""
    cols_to_drop = ["weight", "payer_code", "medical_specialty"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def remove_deceased_patients(df):
    """
    Remove patients who expired during hospital stay.
    Discharge disposition IDs indicating death: 11, 19, 20
    """
    deceased_ids = [11, 19, 20]
    return df[~df["discharge_disposition_id"].isin(deceased_ids)]


def remove_duplicates(df):
    """Remove exact duplicate rows."""
    return df.drop_duplicates()


def map_admission_type(df):
    """Map admission type IDs to human-readable categories."""
    admission_map = {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Not Available",
        6: "NULL",
        7: "Trauma Center",
        8: "Not Mapped"
    }
    df["admission_type"] = df["admission_type_id"].map(admission_map)
    return df


def encode_target_variable(df):
    """Binary encoding: 1 = readmitted, 0 = not readmitted."""
    df["readmitted_binary"] = df["readmitted"].map({
        "NO": 0,
        "<30": 1,
        ">30": 1
    })
    return df


def clean_demographics(df):
    """Clean gender and age columns."""
    df = df[df["gender"].isin(["Male", "Female"])]

    age_order = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    df["age"] = pd.Categorical(df["age"], categories=age_order, ordered=True)
    return df


def encode_medications(df):
    """Convert medication columns to binary indicators."""
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"
    ]

    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)

    df["change"] = df["change"].map({"Ch": 1, "No": 0})
    df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0})

    return df


def create_primary_diabetes_flag(df):
    """Flag primary diagnosis as diabetes (ICD-9 250.xx)."""
    df["primary_diabetes"] = df["diag_1"].astype(str).str.startswith("250").astype(int)
    return df


def final_cleanup(df):
    """Drop remaining NaNs and reset index."""
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def run_cleaning_pipeline():
    """Full data cleaning pipeline."""
    print("Loading data...")
    df = load_data()

    df = drop_high_missing_columns(df)
    df = remove_deceased_patients(df)
    df = remove_duplicates(df)
    df = map_admission_type(df)
    df = encode_target_variable(df)
    df = clean_demographics(df)
    df = encode_medications(df)
    df = create_primary_diabetes_flag(df)
    df = final_cleanup(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Data cleaning complete")
    print(f"Final shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_cleaning_pipeline()
