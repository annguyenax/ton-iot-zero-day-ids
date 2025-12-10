"""
CSV Preprocessing Helper
Cleans dataset CSV files for dashboard upload
"""

import pandas as pd
import numpy as np
import sys
import os

def preprocess_csv(input_file, output_file=None):
    """
    Preprocess CSV file for dashboard upload

    Steps:
    1. Load CSV
    2. Detect and extract label column
    3. Drop non-numeric columns (date, timestamp, categorical strings)
    4. Fill NaN values
    5. Save cleaned CSV
    """

    print(f"\n{'='*60}")
    print(f"Preprocessing: {input_file}")
    print(f"{'='*60}\n")

    # Load CSV
    df = pd.read_csv(input_file)
    print(f"✓ Loaded: {len(df)} rows, {len(df.columns)} columns")

    original_cols = len(df.columns)

    # Detect label column
    label_col = None
    for col in ['label', 'Label', 'attack', 'Attack', 'class', 'Class', 'type', 'Type']:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        print(f"✓ Found label column: '{label_col}'")
        labels = df[label_col]

        # Convert to binary if needed
        if labels.dtype == object:
            print(f"  Converting labels to binary (0=normal, 1=attack)")
            labels = labels.apply(lambda x: 0 if str(x).lower() in ['normal', '0', 'benign', 'legitimate'] else 1)

        # Save labels separately
        df_with_label = df.copy()
        df_with_label[label_col] = labels

        # Remove from features
        df = df.drop(columns=[label_col])
    else:
        print("⚠ No label column found")
        df_with_label = df.copy()

    # Show column types
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())

    # Drop non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()

    if non_numeric_cols:
        print(f"\n⚠ Dropping {len(non_numeric_cols)} non-numeric columns:")
        for col in non_numeric_cols[:10]:  # Show first 10
            sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
            print(f"  - {col}: {sample_val}")
        if len(non_numeric_cols) > 10:
            print(f"  ... and {len(non_numeric_cols)-10} more")

        df = df.drop(columns=non_numeric_cols)

    # Convert all to numeric, coerce errors
    print(f"\nConverting columns to numeric...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check NaN
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()

    if total_nans > 0:
        print(f"\n⚠ Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
        print(f"  Filling with 0...")
        df = df.fillna(0)

    # Drop columns with all zeros or constant values
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"\n⚠ Dropping {len(constant_cols)} constant columns:")
        for col in constant_cols[:10]:
            print(f"  - {col}")
        if len(constant_cols) > 10:
            print(f"  ... and {len(constant_cols)-10} more")
        df = df.drop(columns=constant_cols)

    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Original: {original_cols} columns")
    print(f"After preprocessing: {len(df.columns)} columns")
    print(f"Rows: {len(df)}")
    print(f"Data type: {df.dtypes.unique()}")

    # Save
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"

    # Save with labels
    df_with_label_cleaned = df.copy()
    if label_col:
        df_with_label_cleaned[label_col] = labels

    df_with_label_cleaned.to_csv(output_file, index=False)
    print(f"\n✓ Saved cleaned CSV: {output_file}")
    print(f"  Shape: {df_with_label_cleaned.shape}")

    if label_col:
        label_dist = df_with_label_cleaned[label_col].value_counts()
        print(f"\n  Label distribution:")
        print(f"    Normal (0): {label_dist.get(0, 0)}")
        print(f"    Attack (1): {label_dist.get(1, 0)}")

    print(f"\n{'='*60}")
    print(f"✓ DONE! Upload '{output_file}' to the dashboard")
    print(f"{'='*60}\n")

    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_csv_for_upload.py <input.csv> [output.csv]")
        print("\nExample:")
        print("  python preprocess_csv_for_upload.py ../data/Train_Test_datasets/Train_Test_IoT_dataset/Train_IoT_dataset.csv")
        print("\nThis will create: Train_IoT_dataset_cleaned.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    preprocess_csv(input_file, output_file)
