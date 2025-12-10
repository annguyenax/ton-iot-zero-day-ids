"""
Test Unsupervised Models
Verify zero-day detection capability
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import joblib
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

def test_layer(layer_name):
    """Test a single layer with unsupervised model"""

    print(f"\n{'='*70}")
    print(f"TESTING: {layer_name.upper()}")
    print(f"{'='*70}")

    # Load model
    print("[1/3] Loading model...")
    model = keras.models.load_model(f'../models/unsupervised/{layer_name}_autoencoder.h5', compile=False)
    scaler = joblib.load(f'../models/unsupervised/{layer_name}_scaler.pkl')
    threshold = joblib.load(f'../models/unsupervised/{layer_name}_threshold.pkl')

    print(f"  ✓ Model loaded")
    print(f"  ✓ Threshold: {threshold:.6f}")

    # Load test samples
    print("[2/3] Loading test samples...")
    X_test = np.load(f'../models/unsupervised/{layer_name}_samples_X.npy')
    y_test = np.load(f'../models/unsupervised/{layer_name}_samples_y.npy')

    n_normal = (y_test == 0).sum()
    n_attack = (y_test == 1).sum()

    print(f"  ✓ Loaded {len(X_test)} samples")
    print(f"    Normal: {n_normal}")
    print(f"    Attack (Zero-day): {n_attack}")

    # Predict
    print("[3/3] Running detection...")
    X_scaled = scaler.transform(X_test)
    X_pred = model.predict(X_scaled, verbose=0)
    errors = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

    # Classify
    predictions = (errors > threshold).astype(int)

    # Evaluate
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # Normal samples (should NOT be detected)
    normal_idx = y_test == 0
    normal_errors = errors[normal_idx]
    normal_preds = predictions[normal_idx]

    false_positives = normal_preds.sum()
    true_negatives = len(normal_preds) - false_positives
    false_positive_rate = false_positives / len(normal_preds) if len(normal_preds) > 0 else 0

    print(f"\n📊 NORMAL Samples ({n_normal} samples):")
    print(f"  Correctly classified (True Negative): {true_negatives}")
    print(f"  Incorrectly flagged (False Positive): {false_positives}")
    print(f"  False Positive Rate: {false_positive_rate:.2%}")
    print(f"  Error range: [{normal_errors.min():.6f}, {normal_errors.max():.6f}]")
    print(f"  Error mean: {normal_errors.mean():.6f}")

    # Attack samples (SHOULD be detected as zero-day!)
    attack_idx = y_test == 1
    attack_errors = errors[attack_idx]
    attack_preds = predictions[attack_idx]

    true_positives = attack_preds.sum()
    false_negatives = len(attack_preds) - true_positives
    detection_rate = true_positives / len(attack_preds) if len(attack_preds) > 0 else 0

    print(f"\n🚨 ATTACK Samples (Zero-Day) ({n_attack} samples):")
    print(f"  Correctly detected (True Positive): {true_positives}")
    print(f"  Missed (False Negative): {false_negatives}")
    print(f"  Detection Rate: {detection_rate:.2%}")
    print(f"  Error range: [{attack_errors.min():.6f}, {attack_errors.max():.6f}]")
    print(f"  Error mean: {attack_errors.mean():.6f}")

    # Overall metrics
    accuracy = (true_negatives + true_positives) / len(y_test)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = detection_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📈 OVERALL METRICS:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall (Detection Rate): {recall:.2%}")
    print(f"  F1-Score: {f1:.2%}")

    # Separation
    if len(normal_errors) > 0 and len(attack_errors) > 0:
        separation = attack_errors.mean() / normal_errors.mean()
        print(f"\n🎯 SEPARATION:")
        print(f"  Attack errors / Normal errors: {separation:.2f}x")
        print(f"  Threshold position: {threshold:.6f}")

        if separation > 3.0:
            print(f"  ✓ Excellent separation!")
        elif separation > 2.0:
            print(f"  ✓ Good separation")
        else:
            print(f"  ⚠ Moderate separation - consider adjusting threshold")

    # Show some examples
    print(f"\n📋 EXAMPLES:")

    # Show 3 normal samples
    print(f"\n  Normal samples:")
    for i in range(min(3, n_normal)):
        idx = np.where(normal_idx)[0][i]
        status = "✓ Correctly classified" if predictions[idx] == 0 else "✗ False positive"
        print(f"    Sample {i+1}: Error={errors[idx]:.6f}, Threshold={threshold:.6f} → {status}")

    # Show 3 attack samples
    print(f"\n  Attack samples (Zero-day):")
    for i in range(min(3, n_attack)):
        idx = np.where(attack_idx)[0][i]
        status = "✓ DETECTED!" if predictions[idx] == 1 else "✗ Missed"
        print(f"    Sample {i+1}: Error={errors[idx]:.6f}, Threshold={threshold:.6f} → {status}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': false_positive_rate,
        'detection_rate': detection_rate
    }


def main():
    """Test all layers"""

    print("\n" + "="*70)
    print("UNSUPERVISED MODEL TESTING")
    print("="*70)
    print("\nTesting zero-day detection capability...")
    print("  ✓ Normal samples should NOT trigger alerts")
    print("  ✓ Attack samples should be DETECTED as zero-day")
    print("="*70)

    layers = ['network', 'iot', 'linux', 'windows']
    results = {}

    for layer in layers:
        try:
            stats = test_layer(layer)
            results[layer] = stats
        except FileNotFoundError:
            print(f"\n⚠ Model not found for {layer}")
            print(f"  Run: python train_unsupervised.py first")
        except Exception as e:
            print(f"\n❌ Error testing {layer}: {e}")

    if results:
        # Summary
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")

        for layer, stats in results.items():
            print(f"\n{layer.upper()}:")
            print(f"  Detection Rate (Zero-day): {stats['detection_rate']:.2%}")
            print(f"  False Positive Rate: {stats['false_positive_rate']:.2%}")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  F1-Score: {stats['f1']:.2%}")

        # Average
        avg_detection = np.mean([s['detection_rate'] for s in results.values()])
        avg_fpr = np.mean([s['false_positive_rate'] for s in results.values()])
        avg_acc = np.mean([s['accuracy'] for s in results.values()])

        print(f"\n{'='*70}")
        print(f"AVERAGE ACROSS ALL LAYERS:")
        print(f"  Detection Rate: {avg_detection:.2%}")
        print(f"  False Positive Rate: {avg_fpr:.2%}")
        print(f"  Accuracy: {avg_acc:.2%}")

        if avg_detection > 0.9 and avg_fpr < 0.05:
            print(f"\n✅ EXCELLENT! Model performs very well on zero-day detection!")
        elif avg_detection > 0.8 and avg_fpr < 0.1:
            print(f"\n✅ GOOD! Model has strong zero-day detection capability!")
        elif avg_detection > 0.7:
            print(f"\n⚠ MODERATE. Consider adjusting thresholds.")
        else:
            print(f"\n❌ POOR. Model needs improvement.")

        print(f"{'='*70}")


if __name__ == "__main__":
    main()
