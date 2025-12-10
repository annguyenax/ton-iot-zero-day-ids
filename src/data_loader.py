"""
Data Loading and Exploration Module
Handles loading TON_IoT dataset and initial exploration
"""

import pandas as pd


def load_ton_iot_data(filepath):
    """
    Load TON_IoT dataset from CSV file

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with loaded data
    """
    print(f"[INFO] Loading dataset from: {filepath}")

    # Read CSV with low_memory=False to avoid warnings
    df = pd.read_csv(filepath, low_memory=False)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)[:10]}...")

    return df


def explore_dataset(df):
    """
    Explore dataset structure and find label column

    Args:
        df: Input DataFrame

    Returns:
        str: Name of the label column
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)

    print(f"\nShape: {df.shape}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum().sum()} total")

    # Find label columns
    label_cols = [col for col in df.columns if col.lower() in ['label', 'type', 'attack_type']]

    if not label_cols:
        label_cols = [col for col in df.columns if 'label' in col.lower() or 'type' in col.lower()]

    print(f"\nLabel columns found: {label_cols}")

    if label_cols:
        # Priority: 'label' > 'type' > others
        if 'label' in label_cols:
            label_col = 'label'
        elif 'type' in label_cols:
            label_col = 'type'
        else:
            # Find object/string dtype column
            for col in label_cols:
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    label_col = col
                    break
            else:
                label_col = label_cols[0]

        print(f"\nUsing label column: '{label_col}'")
        print(f"Attack distribution in '{label_col}':")
        print(df[label_col].value_counts())
        print(f"\nData type of '{label_col}': {df[label_col].dtype}")
        return label_col

    return None
