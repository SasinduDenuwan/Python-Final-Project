import pandas as pd
import numpy as np
import os

# ---------------- PATH CONFIGURATION ---------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "diabetic_data.csv")
ID_MAP_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "IDs_mapping.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "diabetic_data_clean.csv")

# ---------------- FUNCTIONS ---------------- #

def load_data(path=RAW_DATA_PATH):
    """Load CSV and convert '?' to NaN"""
    df = pd.read_csv(path, na_values=["?"])
    print(f"Loaded data with shape: {df.shape}")
    return df

def save_clean_data(df, path=OUTPUT_PATH):
    """Save cleaned DataFrame to CSV"""
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}")

def drop_columns_if_missing(df, threshold=0.9, columns=None):
    """Drop columns exceeding threshold of missing values"""
    if columns is None:
        columns = df.columns
    cols_to_drop = [col for col in columns if df[col].isna().mean() > threshold]
    if cols_to_drop:
        print(f"Dropping columns due to missing values > {threshold*100}%: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df

def remove_expired_patients(df, expired_ids=[11]):
    """Remove patients with expired discharge_disposition_id"""
    if 'discharge_disposition_id' not in df.columns:
        print("Column 'discharge_disposition_id' not found.")
        return df
    initial_count = len(df)
    df = df[~df['discharge_disposition_id'].isin(expired_ids)]
    print(f"Removed {initial_count - len(df)} expired patients.")
    return df

def merge_id_descriptions(df, mapping_df, id_col, new_col_name):
    """
    Merge ID descriptions into main DataFrame safely, preventing row duplication.
    """
    # Keep unique ID mappings only
    mapping_df = mapping_df[['id', 'description']].drop_duplicates(subset='id')

    # Merge
    df = df.merge(mapping_df, how='left', left_on=id_col, right_on='id')
    df = df.drop(columns=['id'])
    df = df.rename(columns={'description': new_col_name})
    print(f"Merged descriptions for '{id_col}' into '{new_col_name}' column. Row count: {len(df)}")
    return df

def load_mapping_sections(mapping_csv_path):
    """
    Parse IDs_mapping.csv which contains multiple mapping tables
    separated by header rows and blank lines.
    """

    sections = {
        "admission_type_id": [],
        "discharge_disposition_id": [],
        "admission_source_id": []
    }

    current_section = None

    with open(mapping_csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Detect section headers
            if line.startswith("admission_type_id"):
                current_section = "admission_type_id"
                continue
            elif line.startswith("discharge_disposition_id"):
                current_section = "discharge_disposition_id"
                continue
            elif line.startswith("admission_source_id"):
                current_section = "admission_source_id"
                continue

            # Skip malformed lines
            if "," not in line or current_section is None:
                continue

            id_val, desc = line.split(",", 1)

            if id_val.isdigit():
                sections[current_section].append(
                    {"id": int(id_val), "description": desc.strip('"')}
                )

    # Convert to DataFrames
    admission_type_df = pd.DataFrame(sections["admission_type_id"])
    discharge_df = pd.DataFrame(sections["discharge_disposition_id"])
    admission_source_df = pd.DataFrame(sections["admission_source_id"])

    return admission_type_df, discharge_df, admission_source_df

# ---------------- PIPELINE ---------------- #

def clean_diabetic_data():
    df = load_data()


    print(f"Initial shape: {df.shape}")
    df = drop_columns_if_missing(df, threshold=0.9, columns=['weight'])
    print(f"After dropping weight: {df.shape}")

    df = remove_expired_patients(df, expired_ids=[11,19,20,21])
    print(f"After removing expired patients: {df.shape}")

    admission_type_map, discharge_map, admission_source_map = load_mapping_sections(ID_MAP_PATH)

    # Merge mapping descriptions
    df = merge_id_descriptions(df, admission_type_map, 'admission_type_id', 'admission_type_desc')
    df = merge_id_descriptions(df, discharge_map, 'discharge_disposition_id', 'discharge_desc')
    df = merge_id_descriptions(df, admission_source_map, 'admission_source_id', 'admission_source_desc')

    df_before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed {df_before - df.shape[0]} duplicate rows")


    save_clean_data(df)
    return df

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    cleaned_df = clean_diabetic_data()
    print(cleaned_df.head())