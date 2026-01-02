import pandas as pd
import numpy as np
import os

# --- PATH HANDLING ---
# This ensures the script finds the data regardless of where you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "diabetic_data.csv")

def clean_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find file at {DATA_PATH}")
        return

    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # 2. Handle missing value indicators (Standardizing '?' to NaN)
    df.replace(["?", "NULL", "Not Available", "Unknown/Invalid"], np.nan, inplace=True)

    # 3. Drop columns with too many missing values (Weight is ~97% missing)
    # If we don't drop these, dropna() will delete almost every row in the dataset.
    cols_to_drop = ["weight", "payer_code", "medical_specialty"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # 4. Remove duplicate patients (keep only their first visit)
    df = df.drop_duplicates(subset="patient_nbr", keep="first")

    # 5. Map Admission Type IDs to Descriptions
    admission_type_map = {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Not Available",
        6: "NULL",
        7: "Trauma Center",
        8: "Not Mapped"
    }
    if "admission_type_id" in df.columns:
        df["admission_type"] = df["admission_type_id"].map(admission_type_map)
        df.drop(columns=["admission_type_id"], inplace=True)

    # 6. Target Variable: Binary encoding (1 if readmitted at all, 0 if NO)
    df["readmitted"] = df["readmitted"].map({"NO": 0, "<30": 1, ">30": 1})

    # 7. Clean Gender (Remove the 3 'Unknown/Invalid' records in the dataset)
    df = df[df["gender"].isin(["Male", "Female"])]

    # 8. Define Age as an Ordered Categorical
    age_order = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    df["age"] = pd.Categorical(df["age"], categories=age_order, ordered=True)

    # 9. Binary Encoding for 'change' and 'diabetesMed'
    # Data check: 'change' uses 'Ch' and 'No'; 'diabetesMed' uses 'Yes' and 'No'
    df["change"] = df["change"].map({"Ch": 1, "No": 0})
    df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0})

    # 10. Medication Features: Convert to Binary (1 if Up/Down/Steady, 0 if No)
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", 
        "glimepiride", "acetohexamide", "glipizide", "glyburide", 
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", 
        "miglitol", "troglitazone", "tolazamide", "examide", 
        "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", 
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)

    # 11. Create 'primary_diabetes' feature from diag_1
    # ICD-9 codes 250.xx represent Diabetes Mellitus
    def check_diabetes(code):
        try:
            # Extract first 3 digits of the code
            prefix = str(code)[:3]
            return 1 if prefix == "250" else 0
        except:
            return 0

    df["primary_diabetes"] = df["diag_1"].apply(check_diabetes)

    # 12. Final Clean up: Remove rows with remaining NaNs (e.g., in 'race' or 'diag_1')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 13. Save the cleaned data
    output_path = os.path.join(SCRIPT_DIR, "..", "data", "processed", "cleaned_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Success! Cleaned data saved to: {output_path}")
    print(f"Final dataset shape: {df.shape}")

if __name__ == "__main__":
    clean_data()