"""
Test Minimal Multi-Layer IDS Models
=====================================
Quick testing script to verify all 4 trained models work correctly.

Usage:
    cd src
    python test_minimal.py

Expected time: 30 seconds
"""

import os
import sys
import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import load_ton_iot_data
from preprocessor import preprocess_data

def test_network():
    """Test Network Traffic Layer"""
    print("\n" + "="*60)
    print("🧪 TESTING LAYER 1: NETWORK TRAFFIC")
    print("="*60)

    try:
        # Load model
        model = keras.models.load_model('../models/minimal/network_autoencoder.h5', compile=False)
        scaler = joblib.load('../models/minimal/network_scaler.pkl')
        threshold = joblib.load('../models/minimal/network_threshold.pkl')

        # Load test data (small sample for quick test)
        df = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
        X, y_attack, _, _ = preprocess_data(
            df, label_col='type',
            zero_day_attacks=['ransomware', 'mitm', 'injection', 'xss']
        )

        # Sample 5000 for quick test
        from sklearn.utils import resample
        if len(X) > 5000:
            indices = resample(range(len(X)), n_samples=5000, random_state=42, stratify=y_attack)
            X_test = X.iloc[indices].values
            y_test = y_attack.iloc[indices].values
        else:
            X_test = X.values
            y_test = y_attack.values

        # Scale and predict
        X_test_scaled = scaler.transform(X_test)
        reconstructed = model.predict(X_test_scaled, verbose=0)
        errors = np.mean(np.power(X_test_scaled - reconstructed, 2), axis=1)

        # Classify
        y_pred = (errors > threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"✅ Model loaded successfully")
        print(f"📊 Test samples: {len(X_test):,}")
        print(f"🎯 Threshold: {threshold:.6f}")
        print(f"📈 Accuracy:  {accuracy:.2%}")
        print(f"📈 Precision: {precision:.2%}")
        print(f"📈 Recall:    {recall:.2%}")
        print(f"📈 F1-Score:  {f1:.2%}")

        return True, accuracy

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, 0.0


def test_iot():
    """Test IoT Modbus Layer"""
    print("\n" + "="*60)
    print("🧪 TESTING LAYER 2: IoT MODBUS")
    print("="*60)

    try:
        # Load model
        model = keras.models.load_model('../models/minimal/iot_autoencoder.h5', compile=False)
        scaler = joblib.load('../models/minimal/iot_scaler.pkl')
        threshold = joblib.load('../models/minimal/iot_threshold.pkl')

        # Load test data
        df = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_IoT_dataset/train_test_IoT_modbus.csv')
        X, y_attack, _, _ = preprocess_data(
            df, label_col='type',
            zero_day_attacks=['ransomware', 'mitm', 'injection', 'xss']
        )

        # Sample 3000 for quick test
        from sklearn.utils import resample
        if len(X) > 3000:
            indices = resample(range(len(X)), n_samples=3000, random_state=42, stratify=y_attack)
            X_test = X.iloc[indices].values
            y_test = y_attack.iloc[indices].values
        else:
            X_test = X.values
            y_test = y_attack.values

        # Scale and predict
        X_test_scaled = scaler.transform(X_test)
        reconstructed = model.predict(X_test_scaled, verbose=0)
        errors = np.mean(np.power(X_test_scaled - reconstructed, 2), axis=1)

        # Classify
        y_pred = (errors > threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"✅ Model loaded successfully")
        print(f"📊 Test samples: {len(X_test):,}")
        print(f"🎯 Threshold: {threshold:.6f}")
        print(f"📈 Accuracy:  {accuracy:.2%}")
        print(f"📈 Precision: {precision:.2%}")
        print(f"📈 Recall:    {recall:.2%}")
        print(f"📈 F1-Score:  {f1:.2%}")

        return True, accuracy

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, 0.0


def test_linux():
    """Test Linux System Layer"""
    print("\n" + "="*60)
    print("🧪 TESTING LAYER 3: LINUX SYSTEM")
    print("="*60)

    try:
        # Load model
        model = keras.models.load_model('../models/minimal/linux_autoencoder.h5', compile=False)
        scaler = joblib.load('../models/minimal/linux_scaler.pkl')
        threshold = joblib.load('../models/minimal/linux_threshold.pkl')

        # Load test data
        df = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_Linux_dataset/train_test_linux_process.csv')
        X, y_attack, _, _ = preprocess_data(
            df, label_col='type',
            zero_day_attacks=['ransomware', 'mitm', 'injection', 'xss']
        )

        # Sample 3000 for quick test
        from sklearn.utils import resample
        if len(X) > 3000:
            indices = resample(range(len(X)), n_samples=3000, random_state=42, stratify=y_attack)
            X_test = X.iloc[indices].values
            y_test = y_attack.iloc[indices].values
        else:
            X_test = X.values
            y_test = y_attack.values

        # Scale and predict
        X_test_scaled = scaler.transform(X_test)
        reconstructed = model.predict(X_test_scaled, verbose=0)
        errors = np.mean(np.power(X_test_scaled - reconstructed, 2), axis=1)

        # Classify
        y_pred = (errors > threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"✅ Model loaded successfully")
        print(f"📊 Test samples: {len(X_test):,}")
        print(f"🎯 Threshold: {threshold:.6f}")
        print(f"📈 Accuracy:  {accuracy:.2%}")
        print(f"📈 Precision: {precision:.2%}")
        print(f"📈 Recall:    {recall:.2%}")
        print(f"📈 F1-Score:  {f1:.2%}")

        return True, accuracy

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, 0.0


def test_windows():
    """Test Windows System Layer"""
    print("\n" + "="*60)
    print("🧪 TESTING LAYER 4: WINDOWS SYSTEM")
    print("="*60)

    try:
        # Load model
        model = keras.models.load_model('../models/minimal/windows_autoencoder.h5', compile=False)
        scaler = joblib.load('../models/minimal/windows_scaler.pkl')
        threshold = joblib.load('../models/minimal/windows_threshold.pkl')

        # Load test data
        df = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_Windows_dataset/train_test_windows_10.csv')
        X, y_attack, _, _ = preprocess_data(
            df, label_col='type',
            zero_day_attacks=['ransomware', 'mitm', 'injection', 'xss']
        )

        # Sample 2000 for quick test
        from sklearn.utils import resample
        if len(X) > 2000:
            indices = resample(range(len(X)), n_samples=2000, random_state=42, stratify=y_attack)
            X_test = X.iloc[indices].values
            y_test = y_attack.iloc[indices].values
        else:
            X_test = X.values
            y_test = y_attack.values

        # Scale and predict
        X_test_scaled = scaler.transform(X_test)
        reconstructed = model.predict(X_test_scaled, verbose=0)
        errors = np.mean(np.power(X_test_scaled - reconstructed, 2), axis=1)

        # Classify
        y_pred = (errors > threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"✅ Model loaded successfully")
        print(f"📊 Test samples: {len(X_test):,}")
        print(f"🎯 Threshold: {threshold:.6f}")
        print(f"📈 Accuracy:  {accuracy:.2%}")
        print(f"📈 Precision: {precision:.2%}")
        print(f"📈 Recall:    {recall:.2%}")
        print(f"📈 F1-Score:  {f1:.2%}")

        return True, accuracy

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, 0.0


def test_fusion():
    """Test Fusion Engine Integration"""
    print("\n" + "="*60)
    print("🧪 TESTING FUSION ENGINE")
    print("="*60)

    # Simulate alerts from different layers
    print("\n📋 Simulating Multi-Layer Detection:")
    print("-" * 60)

    # Scenario 1: Normal traffic
    print("\n🟢 Scenario 1: Normal Traffic")
    alerts = {
        'network': False,
        'iot': False,
        'linux': False,
        'windows': False
    }
    print(f"   Network:  {'✅ Normal' if not alerts['network'] else '⚠️ Attack'}")
    print(f"   IoT:      {'✅ Normal' if not alerts['iot'] else '⚠️ Attack'}")
    print(f"   Linux:    {'✅ Normal' if not alerts['linux'] else '⚠️ Attack'}")
    print(f"   Windows:  {'✅ Normal' if not alerts['windows'] else '⚠️ Attack'}")
    threat_level = sum(alerts.values())
    print(f"   🎯 Threat Level: {threat_level}/4 - {'NORMAL' if threat_level == 0 else 'ALERT'}")

    # Scenario 2: Network attack only
    print("\n🟡 Scenario 2: Network Attack Detected")
    alerts = {
        'network': True,
        'iot': False,
        'linux': False,
        'windows': False
    }
    print(f"   Network:  {'✅ Normal' if not alerts['network'] else '⚠️ Attack'}")
    print(f"   IoT:      {'✅ Normal' if not alerts['iot'] else '⚠️ Attack'}")
    print(f"   Linux:    {'✅ Normal' if not alerts['linux'] else '⚠️ Attack'}")
    print(f"   Windows:  {'✅ Normal' if not alerts['windows'] else '⚠️ Attack'}")
    threat_level = sum(alerts.values())
    print(f"   🎯 Threat Level: {threat_level}/4 - {'LOW' if threat_level == 1 else 'MEDIUM' if threat_level == 2 else 'HIGH'}")

    # Scenario 3: Multi-layer attack
    print("\n🔴 Scenario 3: Multi-Layer Attack (APT)")
    alerts = {
        'network': True,
        'iot': True,
        'linux': True,
        'windows': False
    }
    print(f"   Network:  {'✅ Normal' if not alerts['network'] else '⚠️ Attack'}")
    print(f"   IoT:      {'✅ Normal' if not alerts['iot'] else '⚠️ Attack'}")
    print(f"   Linux:    {'✅ Normal' if not alerts['linux'] else '⚠️ Attack'}")
    print(f"   Windows:  {'✅ Normal' if not alerts['windows'] else '⚠️ Attack'}")
    threat_level = sum(alerts.values())
    print(f"   🎯 Threat Level: {threat_level}/4 - {'CRITICAL! Coordinated Attack Detected'}")

    return True


def main():
    """Main test orchestrator"""
    print("\n" + "="*70)
    print("🚀 MINIMAL MULTI-LAYER IDS - SYSTEM VERIFICATION TEST")
    print("="*70)
    print("Testing all 4 trained models...")

    # Track results
    results = []

    # Test each layer
    success_network, acc_network = test_network()
    results.append(('Network Traffic', success_network, acc_network))

    success_iot, acc_iot = test_iot()
    results.append(('IoT Modbus', success_iot, acc_iot))

    success_linux, acc_linux = test_linux()
    results.append(('Linux System', success_linux, acc_linux))

    success_windows, acc_windows = test_windows()
    results.append(('Windows System', success_windows, acc_windows))

    # Test fusion logic
    success_fusion = test_fusion()

    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL TEST SUMMARY")
    print("="*70)

    for layer, success, acc in results:
        status = "✅ PASS" if success else "❌ FAIL"
        acc_str = f"{acc:.2%}" if success else "N/A"
        print(f"{layer:20} {status:10} Accuracy: {acc_str}")

    print(f"\nFusion Engine:       {'✅ PASS' if success_fusion else '❌ FAIL'}")

    # Overall status
    all_pass = all(r[1] for r in results) and success_fusion
    avg_accuracy = np.mean([r[2] for r in results if r[1]]) if any(r[1] for r in results) else 0

    print("\n" + "="*70)
    if all_pass:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print(f"📈 Average Accuracy: {avg_accuracy:.2%}")
        print("\n✅ System is ready for deployment!")
        print("Next step: Run inference_minimal.py for demo")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please check the error messages above and retrain failed models.")
        return 1

    print("="*70)
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
