"""
TON_IoT Zero-day Attack Detection using Autoencoder
Main Training Pipeline - Orchestrates all training modules
"""

import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from data_loader import load_ton_iot_data, explore_dataset
from preprocessor import preprocess_data, create_zero_day_split, normalize_data
from model_builder import build_autoencoder, train_autoencoder
from threshold_finder import find_threshold
from evaluator import evaluate_model, plot_results


def main():
    """Main training pipeline orchestrator"""

    print("="*60)
    print("TON_IoT ZERO-DAY DETECTION - TRAINING PIPELINE")
    print("="*60)

    # ========== CONFIGURATION ==========
    # Change the file path to your dataset location:
    DATA_PATH = "../data/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv"

    # Zero-day attack types (attacks not seen during training)
    ZERO_DAY_ATTACKS = ['ransomware', 'mitm', 'injection', 'xss']

    # Model architecture
    ENCODING_DIMS = [128, 64, 32]  # Encoder layers: 41 → 128 → 64 → 32

    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 256

    # ========== STEP 1: LOAD DATA ==========
    print("\n[STEP 1/8] Loading dataset...")
    df = load_ton_iot_data(DATA_PATH)

    # Check for potential label columns
    print("\n[DEBUG] Checking potential label columns:")
    potential_labels = ['label', 'type', 'attack_type', 'Label', 'Type']
    for col in potential_labels:
        if col in df.columns:
            print(f"\n'{col}' column found:")
            print(f"  Data type: {df[col].dtype}")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Sample values: {df[col].value_counts().head()}")

    label_col = explore_dataset(df)

    # ========== STEP 2: PREPROCESS DATA ==========
    print("\n[STEP 2/8] Preprocessing data...")
    X, y_attack, y_zero_day, y_labels = preprocess_data(
        df,
        label_col=label_col,
        zero_day_attacks=ZERO_DAY_ATTACKS
    )

    # ========== STEP 3: SPLIT DATA ==========
    print("\n[STEP 3/8] Splitting data for zero-day scenario...")
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_zero_day = create_zero_day_split(
        X, y_attack, y_zero_day
    )

    # ========== STEP 4: NORMALIZE DATA ==========
    print("\n[STEP 4/8] Normalizing features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(
        X_train, X_val, X_test
    )

    # ========== STEP 5: BUILD MODEL ==========
    print("\n[STEP 5/8] Building autoencoder model...")
    input_dim = X_train_scaled.shape[1]
    model = build_autoencoder(input_dim, encoding_dims=ENCODING_DIMS)

    # ========== STEP 6: TRAIN MODEL ==========
    print("\n[STEP 6/8] Training autoencoder...")
    history = train_autoencoder(
        model,
        X_train_scaled,
        X_val_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # ========== STEP 7: FIND OPTIMAL THRESHOLD ==========
    print("\n[STEP 7/8] Finding optimal detection threshold...")
    threshold = find_threshold(model, X_train_scaled, X_val_scaled, y_val)

    # ========== STEP 8: EVALUATE MODEL ==========
    print("\n[STEP 8/8] Evaluating model on test set...")
    y_pred, test_errors = evaluate_model(
        model,
        X_test_scaled,
        y_test,
        y_test_zero_day,
        threshold,
        y_labels.iloc[X_test.index] if hasattr(X_test, 'index') else None
    )

    # ========== VISUALIZE RESULTS ==========
    print("\n[INFO] Generating visualizations...")
    plot_results(history, test_errors, y_test, threshold)

    # ========== SAVE ARTIFACTS ==========
    print("\n[INFO] Saving model and artifacts...")

    # Create models directory
    os.makedirs('../models', exist_ok=True)

    # Save model
    model.save('../models/ton_iot_autoencoder.h5')
    print("[INFO] Model saved to '../models/ton_iot_autoencoder.h5'")

    # Save scaler and threshold
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(threshold, '../models/threshold.pkl')
    print("[INFO] Scaler and threshold saved to '../models/'")

    # Save test data for demo and inference
    os.makedirs('../data', exist_ok=True)
    np.save('../data/test_data.npy', X_test_scaled)
    np.save('../data/test_labels.npy', y_test)
    np.save('../data/test_zero_day.npy', y_test_zero_day)
    print("[INFO] Test data saved to '../data/' for demo")

    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nArtifacts saved:")
    print("  - Model: ../models/ton_iot_autoencoder.h5")
    print("  - Scaler: ../models/scaler.pkl")
    print(f"  - Threshold: ../models/threshold.pkl (value: {threshold:.6f})")
    print("  - Test data: ../data/test_*.npy")
    print("  - Visualization: ../results/ton_iot_results.png")
    print("\nNext steps:")
    print("  1. Run inference: cd src && python inference.py")
    print("  2. Run real-time demo: run_realtime.bat (Windows) or python src/realtime_simple.py")
    print("="*60)


if __name__ == "__main__":
    main()
