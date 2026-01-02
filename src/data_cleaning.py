# data_cleaning.py

import pandas as pd
import numpy as np

def load_data(data_path='diabetic_data.csv', mapping_path='IDs_mapping.csv', na_values='?'):
    """
    Load the diabetic data and replace '?' with NaN.
    """
    df = pd.read_csv(data_path, na_values=na_values)
    mapping_df = pd.read_csv(mapping_path)
    return df, mapping_df

def assess_data(df):
    """
    Perform initial assessment: info, describe, head, columns.
    """
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.columns)
    # Convert potential categorical columns
    categorical_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    df[categorical_cols] = df[categorical_cols].astype('category')
    return df

def handle_missing_weight(df, threshold=0.9):
    """
    Check missingness in weight column and drop if >90%.
    """
    missing_ratio = df['weight'].isna().mean()
    print(f"Weight missing ratio: {missing_ratio}")
    if missing_ratio > threshold:
        df = df.drop(columns=['weight'])
        print("Dropped 'weight' column due to high missingness.")
    return df

def filter_deceased_patients(df, mapping_df):
    """
    Identify and filter out deceased patients using discharge_disposition_id.
    """
    # Extract discharge mapping
    discharge_mapping = mapping_df[mapping_df['admission_type_id'].isna() & mapping_df['description'].notna()]
    expired_codes = discharge_mapping[discharge_mapping['description'].str.contains('Expired', na=False)]['discharge_disposition_id'].dropna().astype(int).tolist()
    print(f"Expired codes: {expired_codes}")
    df = df[~df['discharge_disposition_id'].isin(expired_codes)]
    print(f"Filtered out deceased patients.")
    return df

def remove_duplicates(df):
    """
    Remove exact duplicate rows.
    """
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    df = df.drop_duplicates()
    return df

def map_ids_to_descriptions(df, mapping_df):
    """
    Map admission_type_id, discharge_disposition_id, admission_source_id to descriptions.
    """
    # Split mapping_df into sections
    admission_type = mapping_df[mapping_df['admission_type_id'].notna()][['admission_type_id', 'description']].dropna(how='all')
    discharge_disp = mapping_df[mapping_df['discharge_disposition_id'].notna()][['discharge_disposition_id', 'description']].dropna(how='all')
    admission_source = mapping_df[mapping_df['admission_source_id'].notna()][['admission_source_id', 'description']].dropna(how='all')
    
    # Clean and convert
    admission_type['admission_type_id'] = pd.to_numeric(admission_type['admission_type_id'], errors='coerce')
    discharge_disp['discharge_disposition_id'] = pd.to_numeric(discharge_disp['discharge_disposition_id'], errors='coerce')
    admission_source['admission_source_id'] = pd.to_numeric(admission_source['admission_source_id'], errors='coerce')
    
    admission_type = admission_type.dropna(subset=['admission_type_id'])
    discharge_disp = discharge_disp.dropna(subset=['discharge_disposition_id'])
    admission_source = admission_source.dropna(subset=['admission_source_id'])
    
    # Create dicts
    adm_type_dict = dict(zip(admission_type['admission_type_id'], admission_type['description']))
    disp_dict = dict(zip(discharge_disp['discharge_disposition_id'], discharge_disp['description']))
    adm_source_dict = dict(zip(admission_source['admission_source_id'], admission_source['description']))
    
    # Map to new columns
    df['admission_type_desc'] = df['admission_type_id'].map(adm_type_dict)
    df['discharge_disposition_desc'] = df['discharge_disposition_id'].map(disp_dict)
    df['admission_source_desc'] = df['admission_source_id'].map(adm_source_dict)
    
    return df

def clean_data(data_path='diabetic_data.csv', mapping_path='IDs_mapping.csv'):
    df, mapping_df = load_data(data_path, mapping_path)
    df = assess_data(df)
    df = handle_missing_weight(df)
    df = filter_deceased_patients(df, mapping_df)
    df = remove_duplicates(df)
    df = map_ids_to_descriptions(df, mapping_df)
    # Save cleaned data
    df.to_csv('cleaned_diabetic_data.csv', index=False)
    return df

if __name__ == "__main__":
    clean_data()