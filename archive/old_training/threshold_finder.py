"""
Threshold Optimization Module
Finds optimal detection threshold using multiple methods
"""

import numpy as np
from sklearn.metrics import roc_curve


def calculate_reconstruction_error(model, X):
    """
    Calculate reconstruction error (MSE) for each sample

    Args:
        model: Trained autoencoder model
        X: Input features (scaled)

    Returns:
        Array of reconstruction errors
    """
    predictions = model.predict(X, verbose=0)
    mse = np.mean(np.power(X - predictions, 2), axis=1)
    return mse


def find_threshold(model, X_train, X_val, y_val, percentile=95, target_fpr=0.05):
    """
    Find optimal detection threshold using 5 different methods

    Args:
        model: Trained autoencoder model
        X_train: Training features (normal samples)
        X_val: Validation features
        y_val: Validation labels (0=normal, 1=attack)
        percentile: Percentile for method 1
        target_fpr: Target false positive rate for method 4

    Returns:
        Best threshold value
    """
    print("\n" + "="*60)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*60)

    # Calculate errors on train (normal samples)
    train_errors = calculate_reconstruction_error(model, X_train)

    # Method 1: Percentile
    threshold_percentile = np.percentile(train_errors, percentile)

    # Method 2: Mean + k*std
    threshold_std = np.mean(train_errors) + 3 * np.std(train_errors)

    # Method 3: ROC curve on validation set
    val_errors = calculate_reconstruction_error(model, X_val)
    fpr, tpr, thresholds = roc_curve(y_val, val_errors)

    # Find threshold with highest J-statistic (Youden's index: TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    threshold_roc = thresholds[optimal_idx]

    # Method 4: Target FPR (e.g., FPR <= 5%)
    fpr_mask = fpr <= target_fpr
    if fpr_mask.any():
        valid_indices = np.where(fpr_mask)[0]
        best_tpr_idx = valid_indices[np.argmax(tpr[fpr_mask])]
        threshold_fpr_target = thresholds[best_tpr_idx]
    else:
        threshold_fpr_target = threshold_percentile

    # Method 5: Balanced threshold - target 10% FPR for better TPR
    fpr_mask_balanced = fpr <= 0.10
    if fpr_mask_balanced.any():
        valid_indices = np.where(fpr_mask_balanced)[0]
        best_tpr_idx = valid_indices[np.argmax(tpr[fpr_mask_balanced])]
        threshold_balanced = thresholds[best_tpr_idx]
    else:
        threshold_balanced = threshold_roc

    print(f"\nThreshold methods:")
    print(f"  1. Percentile {percentile}%: {threshold_percentile:.6f}")
    print(f"  2. Mean + 3*STD: {threshold_std:.6f}")
    print(f"  3. ROC optimal (max J-score): {threshold_roc:.6f}")
    print(f"  4. Target FPR<={target_fpr*100}%: {threshold_fpr_target:.6f}")
    print(f"  5. Balanced (FPR<=10%): {threshold_balanced:.6f}")

    # Test each threshold on validation set
    print(f"\nValidation set performance:")
    thresholds_to_test = [
        ("Percentile", threshold_percentile),
        ("Mean+3STD", threshold_std),
        ("ROC optimal", threshold_roc),
        ("FPR Target 5%", threshold_fpr_target),
        ("Balanced 10%", threshold_balanced)
    ]

    best_f1 = 0
    best_threshold = threshold_roc
    best_name = "ROC optimal"

    for name, thresh in thresholds_to_test:
        y_pred = (val_errors > thresh).astype(int)
        val_fpr = np.sum((y_pred == 1) & (y_val == 0)) / np.sum(y_val == 0)
        val_tpr = np.sum((y_pred == 1) & (y_val == 1)) / np.sum(y_val == 1)

        # Calculate F1 score
        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  {name:20} -> TPR: {val_tpr:6.2%}, FPR: {val_fpr:6.2%}, F1: {f1:.4f}")

        # Select threshold with best F1 and TPR > 85%
        if f1 > best_f1 and val_tpr > 0.85:
            best_f1 = f1
            best_threshold = thresh
            best_name = name

    # If no threshold achieves TPR > 85%, default to ROC optimal
    if best_f1 == 0:
        print("\n[WARNING] No threshold achieves TPR > 85%")
        print("[INFO] Defaulting to ROC optimal for maximum detection")
        best_threshold = threshold_roc
        best_name = "ROC optimal"

    print(f"\n[SELECTED] Using {best_name} threshold: {best_threshold:.6f}")

    # Show selected performance
    y_pred_selected = (val_errors > best_threshold).astype(int)
    val_fpr_selected = np.sum((y_pred_selected == 1) & (y_val == 0)) / np.sum(y_val == 0)
    val_tpr_selected = np.sum((y_pred_selected == 1) & (y_val == 1)) / np.sum(y_val == 1)
    print(f"[INFO] Selected threshold performance: TPR={val_tpr_selected:.2%}, FPR={val_fpr_selected:.2%}")

    return best_threshold
