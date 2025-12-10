"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and train/val/test splitting for zero-day scenario
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df, label_col='label', zero_day_attacks=['ransomware', 'mitm', 'injection']):
    """
    Preprocess dataset for zero-day scenario

    Args:
        df: DataFrame
        label_col: name of label column
        zero_day_attacks: list of attack types to treat as zero-day (not in training)

    Returns:
        X: Feature matrix
        y_attack: Binary labels (0=normal, 1=attack)
        y_zero_day: Zero-day flags
        y_labels: Original labels
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)

    df = df.copy()

    # 1. Handle missing values
    print("[1/7] Handling missing values...")
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna('unknown')

    # 2. Separate features and labels
    print("[2/7] Separating features and labels...")

    if label_col not in df.columns:
        print(f"[WARNING] Label column '{label_col}' not found. Using first column with 'label' in name.")
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if label_cols:
            label_col = label_cols[0]
        else:
            raise ValueError("No label column found!")

    print(f"Using label column: '{label_col}' (dtype: {df[label_col].dtype})")

    # Create binary label (0=normal, 1=attack)
    # Handle both string and numeric labels
    if df[label_col].dtype == 'object' or df[label_col].dtype == 'string':
        # String labels
        df['is_attack'] = (df[label_col].str.lower() != 'normal').astype(int)

        # Mark zero-day attacks
        df['is_zero_day'] = df[label_col].str.lower().apply(
            lambda x: any(zd in str(x).lower() for zd in zero_day_attacks)
        )
    else:
        # Numeric labels - assume 0 = normal, non-zero = attack
        print("[INFO] Numeric labels detected. Assuming 0=normal, non-zero=attack")
        df['is_attack'] = (df[label_col] != 0).astype(int)

        # Cannot distinguish zero-day with numeric labels
        # Need manual mapping or use another column
        print("[WARNING] Cannot identify zero-day attacks from numeric labels.")
        print("[INFO] Treating all attacks as 'known' for training purposes.")
        df['is_zero_day'] = False  # All attacks are known

        # If 'type' column exists, use it
        if 'type' in df.columns and df['type'].dtype == 'object':
            print("[INFO] Found 'type' column - using it for zero-day classification")
            df['is_zero_day'] = df['type'].str.lower().apply(
                lambda x: any(zd in str(x).lower() for zd in zero_day_attacks)
            )

    print(f"Normal samples: {(df['is_attack']==0).sum()}")
    print(f"Attack samples: {(df['is_attack']==1).sum()}")
    print(f"Zero-day samples: {df['is_zero_day'].sum()}")

    # Save label column for later use
    y_labels = df[label_col].copy()

    # Drop label columns
    drop_cols = [label_col, 'is_attack', 'is_zero_day'] + [col for col in df.columns if 'label' in col.lower()]
    X = df.drop(columns=drop_cols, errors='ignore')

    # 3. Encode categorical columns
    print("[3/7] Encoding categorical features...")
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if X[col].nunique() < 100:  # If few categories
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X = X.drop(columns=[col])  # Drop if too many categories

    # 4. Convert all to numeric
    print("[4/7] Converting to numeric...")
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # 5. Remove infinite values
    print("[5/7] Removing infinite values...")
    X = X.replace([np.inf, -np.inf], 0)

    # 6. Feature selection (remove constant features)
    print("[6/7] Removing constant features...")
    constant_cols = X.columns[X.std() == 0]
    X = X.drop(columns=constant_cols)

    print(f"[7/7] Final feature count: {X.shape[1]}")

    return X, df['is_attack'], df['is_zero_day'], y_labels


# REMOVED: create_zero_day_split() function
# This function was designed for supervised learning (train on normal + known attacks)
# It is NOT used in the current unsupervised approach (train_unsupervised.py)
# The unsupervised approach trains ONLY on normal data for true zero-day detection

def normalize_data(X_train, X_val, X_test):
    """
    Normalize data using StandardScaler

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    print("\n[INFO] Normalizing data...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
