"""
TON_IoT Zero-day Detection - Inference & Demo
Load trained model and detect attacks in real-time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import joblib
import warnings

warnings.filterwarnings("ignore")


class ZeroDayDetector:
    """
    Zero-day attack detector using trained autoencoder
    """

    def __init__(
        self,
        model_path="ton_iot_autoencoder.h5",
        scaler_path="scaler.pkl",
        threshold_path="threshold.pkl",
    ):
        """Load trained model and preprocessing objects"""

        print("=" * 60)
        print("LOADING ZERO-DAY DETECTOR")
        print("=" * 60)

        # Load model (compile=False để không bị lỗi metric legacy)
        self.model = keras.models.load_model(model_path, compile=False)
        print(f"[✓] Model loaded from {model_path}")

        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"[✓] Scaler loaded from {scaler_path}")

        # Load threshold
        self.threshold = joblib.load(threshold_path)
        print(f"[✓] Threshold loaded: {self.threshold:.6f}")

        self.detection_history = []

    def preprocess_sample(self, sample):
        """Preprocess a single sample or batch (raw features -> scaled)"""
        if isinstance(sample, pd.Series):
            sample = sample.values.reshape(1, -1)
        elif isinstance(sample, pd.DataFrame):
            sample = sample.values
        else:
            sample = np.asarray(sample)
            if len(sample.shape) == 1:
                sample = sample.reshape(1, -1)

        # Scale với StandardScaler đã train từ train.py
        sample_scaled = self.scaler.transform(sample)
        return sample_scaled

    def calculate_reconstruction_error(self, sample_scaled):
        """Calculate reconstruction error"""
        reconstructed = self.model.predict(sample_scaled, verbose=0)
        mse = np.mean(np.power(sample_scaled - reconstructed, 2), axis=1)
        return mse

    def detect(self, sample, return_details=False, already_scaled=False):
        """
        Detect if sample is anomalous (attack)

        Args:
            sample: 1 sample (1D) hoặc batch (2D)
            return_details: True -> trả thêm info
            already_scaled:
                - False: sample là raw feature -> sẽ được scaler.transform()
                - True : sample đã được scale sẵn (ví dụ test_data.npy) -> không scale nữa
        """

        # Preprocess
        if already_scaled:
            # Đảm bảo shape (n_samples, n_features)
            if isinstance(sample, pd.Series):
                sample_scaled = sample.values.reshape(1, -1)
            elif isinstance(sample, pd.DataFrame):
                sample_scaled = sample.values
            else:
                sample = np.asarray(sample)
                if len(sample.shape) == 1:
                    sample_scaled = sample.reshape(1, -1)
                else:
                    sample_scaled = sample
        else:
            sample_scaled = self.preprocess_sample(sample)

        # Calculate error
        error = self.calculate_reconstruction_error(sample_scaled)[0]

        # Predict
        is_attack = int(error > self.threshold)

        # Confidence (càng xa threshold càng tự tin)
        confidence = abs(error - self.threshold) / self.threshold * 100

        # Lưu history
        self.detection_history.append(
            {"error": error, "prediction": is_attack, "confidence": confidence}
        )

        if return_details:
            return {
                "prediction": "ATTACK" if is_attack else "NORMAL",
                "reconstruction_error": error,
                "threshold": self.threshold,
                "confidence": confidence,
                "severity": self._calculate_severity(error),
            }

        return is_attack, confidence

    def _calculate_severity(self, error):
        """Calculate attack severity based on error magnitude"""
        ratio = error / self.threshold

        if ratio < 1:
            return "NORMAL"
        elif ratio < 1.5:
            return "LOW"
        elif ratio < 2.0:
            return "MEDIUM"
        elif ratio < 3.0:
            return "HIGH"
        else:
            return "CRITICAL"

    def batch_detect(self, X, already_scaled=False):
        """
        Detect attacks in batch

        Args:
            X: array 2D
            already_scaled:
                - False: X là raw feature -> scaler.transform()
                - True : X đã được scale (test_data.npy) -> dùng trực tiếp
        """
        if already_scaled:
            X_scaled = X
        else:
            X_scaled = self.preprocess_sample(X)

        errors = self.calculate_reconstruction_error(X_scaled)
        predictions = (errors > self.threshold).astype(int)

        return predictions, errors

    def get_statistics(self):
        """Get detection statistics"""
        if not self.detection_history:
            return None

        total = len(self.detection_history)
        attacks = sum(1 for h in self.detection_history if h["prediction"] == 1)

        return {
            "total_samples": total,
            "attacks_detected": attacks,
            "normal_samples": total - attacks,
            "attack_rate": attacks / total * 100,
            "avg_error": np.mean([h["error"] for h in self.detection_history]),
            "avg_confidence": np.mean(
                [h["confidence"] for h in self.detection_history]
            ),
        }


# ============================================
# DEMO FUNCTIONS
# ============================================


def demo_real_time_detection(detector, test_data, test_labels, num_samples=1000):
    """
    Demo phát hiện real-time
    test_data: đã được scale (X_test_scaled) -> dùng already_scaled=True
    """
    print("\n" + "=" * 60)
    print("REAL-TIME DETECTION DEMO")
    print("=" * 60)

    # Random samples
    indices = np.random.choice(len(test_data), num_samples, replace=False)

    results = []

    print("\nDetecting attacks in real-time...\n")
    print(
        f"{'#':<5} {'Error':<12} {'Threshold':<12} {'Prediction':<12} "
        f"{'Confidence':<12} {'Severity':<12}"
    )
    print("-" * 75)

    for i, idx in enumerate(indices):
        sample = test_data[idx]
        true_label = "ATTACK" if test_labels[idx] == 1 else "NORMAL"

        # Detect (data đã scale -> already_scaled=True)
        result = detector.detect(sample, return_details=True, already_scaled=True)

        # Color coding
        pred_symbol = "🔴" if result["prediction"] == "ATTACK" else "🟢"
        match = "✓" if result["prediction"] == true_label else "✗"

        print(
            f"{i+1:<5} {result['reconstruction_error']:<12.6f} "
            f"{result['threshold']:<12.6f} {result['prediction']:<12} "
            f"{result['confidence']:<12.2f} {result['severity']:<12} "
            f"{pred_symbol} {match}"
        )

        results.append(
            {
                "true_label": true_label,
                "predicted": result["prediction"],
                "error": result["reconstruction_error"],
                "severity": result["severity"],
            }
        )

    # Statistics
    correct = sum(1 for r in results if r["true_label"] == r["predicted"])
    accuracy = correct / len(results) * 100

    print("\n" + "-" * 75)
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")

    return results


def demo_batch_detection(detector, test_data, test_labels):
    """
    Demo phát hiện batch và visualize
    test_data: đã được scale -> already_scaled=True
    """
    print("\n" + "=" * 60)
    print("BATCH DETECTION DEMO")
    print("=" * 60)

    # Detect all
    predictions, errors = detector.batch_detect(test_data, already_scaled=True)

    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n[CLASSIFICATION REPORT]")
    print(
        classification_report(
            test_labels, predictions, target_names=["Normal", "Attack"]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix")

    # 2. Error distribution
    normal_errors = errors[test_labels == 0]
    attack_errors = errors[test_labels == 1]

    axes[1].hist(normal_errors, bins=50, alpha=0.6, label="Normal", color="green")
    axes[1].hist(attack_errors, bins=50, alpha=0.6, label="Attack", color="red")
    axes[1].axvline(
        detector.threshold,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
    axes[1].set_xlabel("Reconstruction Error")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error Distribution")
    axes[1].legend()
    axes[1].set_yscale("log")

    # 3. Detection over time
    axes[2].scatter(
        range(len(errors)),
        errors,
        c=test_labels,
        cmap="RdYlGn_r",
        alpha=0.5,
        s=10,
    )
    axes[2].axhline(
        detector.threshold,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
    axes[2].set_xlabel("Sample Index")
    axes[2].set_ylabel("Reconstruction Error")
    axes[2].set_title("Detection Results")
    axes[2].legend()

    plt.tight_layout()

    # Save to results directory
    import os
    os.makedirs('../results', exist_ok=True)
    plt.savefig("../results/detection_results.png", dpi=300, bbox_inches="tight")
    print("\n[INFO] Results saved to '../results/detection_results.png'")
    plt.show()

    return predictions, errors


def demo_attack_severity_analysis(detector, test_data, test_labels, max_samples=50000):
    """
    Phân tích mức độ nghiêm trọng của attacks (tối ưu, chạy nhanh)
    - Dùng batch predict thay vì gọi detect() từng sample
    - max_samples: giới hạn số attack mẫu để phân tích cho nhẹ
    """
    print("\n" + "="*60)
    print("ATTACK SEVERITY ANALYSIS")
    print("="*60)

    # Lấy index các mẫu tấn công
    attack_indices = np.where(test_labels == 1)[0]
    n_attacks = len(attack_indices)
    print(f"[INFO] Total attack samples: {n_attacks}")

    # Giới hạn số lượng cho demo (để tránh quá nặng)
    if n_attacks > max_samples:
        print(f"[INFO] Subsampling {max_samples} attacks (from {n_attacks}) for analysis")
        attack_indices = np.random.choice(attack_indices, max_samples, replace=False)

    attack_data = test_data[attack_indices]

    # --- Batch xử lý ---
    # 1) test_data đã được scale sẵn -> dùng trực tiếp
    attack_scaled = attack_data

    # 2) Tính reconstruction error cho toàn bộ
    errors = detector.calculate_reconstruction_error(attack_scaled)

    # 3) Tính severity theo cùng logic với _calculate_severity()
    ratio = errors / detector.threshold

    severities = np.empty(len(errors), dtype=object)
    severities[ratio < 1.0] = "NORMAL"
    severities[(ratio >= 1.0) & (ratio < 1.5)] = "LOW"
    severities[(ratio >= 1.5) & (ratio < 2.0)] = "MEDIUM"
    severities[(ratio >= 2.0) & (ratio < 3.0)] = "HIGH"
    severities[ratio >= 3.0] = "CRITICAL"

    # --- Thống kê ---
    from collections import Counter
    severity_counts = Counter(severities)

    print("\nSeverity Distribution (on attack samples):")
    for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        count = severity_counts.get(severity, 0)
        percentage = count / len(severities) * 100 if len(severities) > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"{severity:<10} {count:>6} ({percentage:>5.1f}%) {bar}")

    # --- Vẽ biểu đồ ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    sizes = [severity_counts.get(s, 0) for s in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']]
    colors = ['#90EE90', '#FFD700', '#FF8C00', '#FF0000']
    axes[0].pie(
        sizes,
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
    )
    axes[0].set_title('Attack Severity Distribution')

    # Boxplot theo severity
    severity_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    severity_errors = {s: [] for s in severity_order}
    for sev, err in zip(severities, errors):
        if sev in severity_errors:
            severity_errors[sev].append(err)

    data_to_plot = [severity_errors[s] for s in severity_order if severity_errors[s]]
    labels_to_plot = [s for s in severity_order if severity_errors[s]]

    if data_to_plot:
        axes[1].boxplot(data_to_plot, labels=labels_to_plot)
        axes[1].set_xlabel('Severity Level')
        axes[1].set_ylabel('Reconstruction Error')
        axes[1].set_title('Error Distribution by Severity')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No attack samples to plot",
                     ha='center', va='center')
        axes[1].axis('off')

    plt.tight_layout()

    # Save to results directory
    import os
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/severity_analysis.png', dpi=300, bbox_inches='tight')
    print("\n[INFO] Severity analysis saved to '../results/severity_analysis.png'")
    plt.show()



def interactive_demo():
    """
    Interactive demo - user nhập features để test
    (ở đây sample là raw feature nên vẫn cần scaler.transform)
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE DETECTION DEMO")
    print("=" * 60)
    print("\nThis mode allows you to input custom network traffic features")
    print("for real-time attack detection.\n")

    detector = ZeroDayDetector(
        model_path="../models/ton_iot_autoencoder.h5",
        scaler_path="../models/scaler.pkl",
        threshold_path="../models/threshold.pkl",
    )

    print("Example usage:")
    print("- Enter 'random' for random test")
    print("- Enter 'scaled:<values>' if data is already scaled (from test_data.npy)")
    print("- Enter '<values>' for raw features (will be scaled automatically)")
    print("- Enter 'q' to quit\n")

    while True:
        user_input = input(
            "Enter features: "
        ).strip()

        if user_input.lower() == "q":
            break

        # Check if data is already scaled
        already_scaled = False
        if user_input.lower().startswith("scaled:"):
            already_scaled = True
            user_input = user_input[7:].strip()  # Remove "scaled:" prefix

        if user_input.lower() == "random":
            sample = np.random.randn(detector.model.input_shape[1])
            print(f"Random sample generated with {len(sample)} features")
            already_scaled = False  # Random data is not scaled
        else:
            try:
                sample = np.array(
                    [float(x.strip()) for x in user_input.split(",")]
                )

                if len(sample) != detector.model.input_shape[1]:
                    print(
                        f"Error: Expected {detector.model.input_shape[1]} "
                        f"features, got {len(sample)}"
                    )
                    continue
            except Exception:
                print("Error: Invalid input format. Use comma-separated numbers.")
                continue

        # Detect
        result = detector.detect(sample, return_details=True, already_scaled=already_scaled)

        print("\n" + "-" * 50)
        if result["prediction"] == "ATTACK":
            print(f"🔴 ALERT: {result['prediction']} DETECTED!")
        else:
            print(f"🟢 Status: {result['prediction']}")

        print(f"Reconstruction Error: {result['reconstruction_error']:.6f}")
        print(f"Threshold: {result['threshold']:.6f}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Severity: {result['severity']}")
        print("-" * 50 + "\n")


# ============================================
# MAIN DEMO
# ============================================


def main():
    """Main demo pipeline"""

    print("=" * 60)
    print("TON_IoT ZERO-DAY DETECTION - DEMO")
    print("=" * 60)

    # Initialize detector with correct paths
    detector = ZeroDayDetector(
        model_path="../models/ton_iot_autoencoder.h5",
        scaler_path="../models/scaler.pkl",
        threshold_path="../models/threshold.pkl",
    )

    # Load test data (ĐÃ SCALE trong train.py)
    print("\n[INFO] Loading test data...")
    test_data = np.load("../data/test_data.npy")
    test_labels = np.load("../data/test_labels.npy")
    test_zero_day = np.load("../data/test_zero_day.npy")

    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Menu
    while True:
        print("\n" + "=" * 60)
        print("DEMO MENU")
        print("=" * 60)
        print("1. Real-time Detection Demo (50 samples)")
        print("2. Batch Detection & Visualization")
        print("3. Attack Severity Analysis")
        print("4. Interactive Demo")
        print("5. Show Statistics")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            demo_real_time_detection(detector, test_data, test_labels, num_samples=50)

        elif choice == "2":
            demo_batch_detection(detector, test_data, test_labels)

        elif choice == "3":
            demo_attack_severity_analysis(detector, test_data, test_labels)

        elif choice == "4":
            interactive_demo()

        elif choice == "5":
            stats = detector.get_statistics()
            if stats:
                print("\n[DETECTION STATISTICS]")
                for key, value in stats.items():
                    print(f"{key}: {value}")
            else:
                print("No detection history yet.")

        elif choice == "6":
            print("\nExiting demo. Goodbye!")
            break

        else:
            print("Invalid choice. Please select 1-6.")


if __name__ == "__main__":
    main()
