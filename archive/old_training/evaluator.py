"""
Model Evaluation Module
Handles performance evaluation, metrics calculation, and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os

from threshold_finder import calculate_reconstruction_error


def evaluate_model(model, X_test, y_test, y_test_zero_day, threshold, y_labels_test=None):
    """
    Evaluate model performance on test set

    Args:
        model: Trained autoencoder model
        X_test: Test features (scaled)
        y_test: Test labels (0=normal, 1=attack)
        y_test_zero_day: Zero-day flags
        threshold: Detection threshold
        y_labels_test: Original test labels (optional)

    Returns:
        y_pred: Predicted labels
        test_errors: Reconstruction errors
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Predict
    test_errors = calculate_reconstruction_error(model, X_test)
    y_pred = (test_errors > threshold).astype(int)

    # Overall metrics
    print("\n[OVERALL PERFORMANCE]")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    detection_rate = tp / (tp + fn) * 100
    false_positive_rate = fp / (fp + tn) * 100

    print(f"\nDetection Rate (Recall): {detection_rate:.2f}%")
    print(f"False Positive Rate: {false_positive_rate:.2f}%")

    # Zero-day specific evaluation
    print("\n[ZERO-DAY DETECTION]")
    zero_day_mask = y_test_zero_day == 1
    known_attack_mask = (y_test == 1) & (y_test_zero_day == 0)

    if zero_day_mask.sum() > 0:
        zero_day_detected = y_pred[zero_day_mask].sum()
        zero_day_total = zero_day_mask.sum()
        zero_day_rate = zero_day_detected / zero_day_total * 100
        print(f"Zero-day attacks detected: {zero_day_detected}/{zero_day_total} ({zero_day_rate:.2f}%)")

    if known_attack_mask.sum() > 0:
        known_detected = y_pred[known_attack_mask].sum()
        known_total = known_attack_mask.sum()
        known_rate = known_detected / known_total * 100
        print(f"Known attacks detected: {known_detected}/{known_total} ({known_rate:.2f}%)")

    return y_pred, test_errors


def plot_results(history, test_errors, y_test, threshold, output_dir='../results'):
    """
    Plot and save visualization of training and evaluation results

    Args:
        history: Training history object
        test_errors: Reconstruction errors on test set
        y_test: True labels
        threshold: Detection threshold
        output_dir: Directory to save plots
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Training loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Reconstruction error distribution
    normal_errors = test_errors[y_test == 0]
    attack_errors = test_errors[y_test == 1]

    axes[0, 1].hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='green')
    axes[0, 1].hist(attack_errors, bins=50, alpha=0.7, label='Attack', color='red')
    axes[0, 1].axvline(threshold, color='blue', linestyle='--', label=f'Threshold: {threshold:.4f}')
    axes[0, 1].set_xlabel('Reconstruction Error')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')

    # 3. Error over samples
    axes[1, 0].scatter(range(len(test_errors)), test_errors, c=y_test, cmap='RdYlGn_r', alpha=0.5, s=1)
    axes[1, 0].axhline(threshold, color='blue', linestyle='--', label='Threshold')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Reconstruction Error')
    axes[1, 0].set_title('Reconstruction Error per Sample')
    axes[1, 0].legend()

    # 4. ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_errors)
    roc_auc = auc(fpr, tpr)

    axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'ton_iot_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Results saved to '{output_path}'")
    plt.show()
