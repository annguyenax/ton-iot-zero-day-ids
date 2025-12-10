"""
Minimal Multi-Layer IDS - Inference Demo
==========================================
Demonstrates the 4-layer detection system with simulated attacks.

Usage:
    cd src
    python inference_minimal.py

Features:
    - Load all 4 trained models
    - Test with normal and attack samples
    - Show layer-by-layer detection
    - Fusion engine decision
    - Severity analysis
"""

import os
import sys
import numpy as np
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import load_ton_iot_data
from preprocessor import preprocess_data


class MultiLayerIDS:
    """Multi-Layer Intrusion Detection System"""

    def __init__(self, models_dir='../models/minimal'):
        """Initialize all 4 layers"""
        print("🔧 Loading Multi-Layer IDS...")
        print("="*60)

        self.models_dir = models_dir
        self.layers = {}

        # Load Network Layer
        try:
            self.layers['network'] = {
                'model': keras.models.load_model(f'{models_dir}/network_autoencoder.h5', compile=False),
                'scaler': joblib.load(f'{models_dir}/network_scaler.pkl'),
                'threshold': joblib.load(f'{models_dir}/network_threshold.pkl'),
                'name': 'Network Traffic',
                'weight': 0.30
            }
            print("✅ Layer 1: Network Traffic (Weight: 30%)")
        except Exception as e:
            print(f"❌ Layer 1 failed: {e}")
            sys.exit(1)

        # Load IoT Layer
        try:
            self.layers['iot'] = {
                'model': keras.models.load_model(f'{models_dir}/iot_autoencoder.h5', compile=False),
                'scaler': joblib.load(f'{models_dir}/iot_scaler.pkl'),
                'threshold': joblib.load(f'{models_dir}/iot_threshold.pkl'),
                'name': 'IoT Modbus',
                'weight': 0.25
            }
            print("✅ Layer 2: IoT Modbus (Weight: 25%)")
        except Exception as e:
            print(f"❌ Layer 2 failed: {e}")
            sys.exit(1)

        # Load Linux Layer
        try:
            self.layers['linux'] = {
                'model': keras.models.load_model(f'{models_dir}/linux_autoencoder.h5', compile=False),
                'scaler': joblib.load(f'{models_dir}/linux_scaler.pkl'),
                'threshold': joblib.load(f'{models_dir}/linux_threshold.pkl'),
                'name': 'Linux System',
                'weight': 0.25
            }
            print("✅ Layer 3: Linux System (Weight: 25%)")
        except Exception as e:
            print(f"❌ Layer 3 failed: {e}")
            sys.exit(1)

        # Load Windows Layer
        try:
            self.layers['windows'] = {
                'model': keras.models.load_model(f'{models_dir}/windows_autoencoder.h5', compile=False),
                'scaler': joblib.load(f'{models_dir}/windows_scaler.pkl'),
                'threshold': joblib.load(f'{models_dir}/windows_threshold.pkl'),
                'name': 'Windows System',
                'weight': 0.20
            }
            print("✅ Layer 4: Windows System (Weight: 20%)")
        except Exception as e:
            print(f"❌ Layer 4 failed: {e}")
            sys.exit(1)

        print("="*60)
        print("✅ All layers loaded successfully!\n")

    def detect_layer(self, layer_key, features):
        """Detect anomaly in a single layer"""
        layer = self.layers[layer_key]

        # Scale features
        scaled = layer['scaler'].transform(features.reshape(1, -1))

        # Reconstruct
        reconstructed = layer['model'].predict(scaled, verbose=0)

        # Calculate error
        error = np.mean(np.power(scaled - reconstructed, 2))

        # Classify
        is_attack = error > layer['threshold']

        # Calculate severity (0-100)
        if is_attack:
            severity = min(100, (error / layer['threshold'] - 1) * 100)
        else:
            severity = 0

        return is_attack, error, severity

    def detect_multi_layer(self, samples):
        """
        Detect attacks using all layers with fusion engine

        Args:
            samples: dict with keys 'network', 'iot', 'linux', 'windows'
                    Each value is a numpy array of features

        Returns:
            dict with detection results for each layer and fusion decision
        """
        results = {}

        for layer_key, features in samples.items():
            if layer_key in self.layers and features is not None:
                is_attack, error, severity = self.detect_layer(layer_key, features)
                results[layer_key] = {
                    'is_attack': is_attack,
                    'error': error,
                    'severity': severity,
                    'threshold': self.layers[layer_key]['threshold']
                }

        # Fusion Engine: Weighted voting
        fusion_score = 0.0
        for layer_key, result in results.items():
            if result['is_attack']:
                fusion_score += self.layers[layer_key]['weight']

        # Decision thresholds
        if fusion_score >= 0.5:
            threat_level = 'CRITICAL'
            color = '🔴'
        elif fusion_score >= 0.3:
            threat_level = 'HIGH'
            color = '🟠'
        elif fusion_score >= 0.1:
            threat_level = 'MEDIUM'
            color = '🟡'
        else:
            threat_level = 'LOW/NORMAL'
            color = '🟢'

        results['fusion'] = {
            'score': fusion_score,
            'threat_level': threat_level,
            'color': color,
            'is_attack': fusion_score >= 0.3
        }

        return results


def load_test_samples():
    """Load test samples for each layer"""
    print("\n📦 Loading test samples...")
    print("="*60)

    samples = {}

    # Network sample
    try:
        df_net = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
        X_net, y_net, _, _ = preprocess_data(df_net, label_col='type', zero_day_attacks=[])

        # Get one normal and one attack sample
        normal_idx = np.where(y_net == 0)[0][0]
        attack_idx = np.where(y_net == 1)[0][0]

        samples['network'] = {
            'normal': X_net.iloc[normal_idx].values,
            'attack': X_net.iloc[attack_idx].values
        }
        print("✅ Network samples loaded")
    except Exception as e:
        print(f"⚠️  Network samples failed: {e}")
        samples['network'] = None

    # IoT sample
    try:
        df_iot = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_IoT_dataset/train_test_IoT_modbus.csv')
        X_iot, y_iot, _, _ = preprocess_data(df_iot, label_col='type', zero_day_attacks=[])

        normal_idx = np.where(y_iot == 0)[0][0]
        attack_idx = np.where(y_iot == 1)[0][0]

        samples['iot'] = {
            'normal': X_iot.iloc[normal_idx].values,
            'attack': X_iot.iloc[attack_idx].values
        }
        print("✅ IoT Modbus samples loaded")
    except Exception as e:
        print(f"⚠️  IoT samples failed: {e}")
        samples['iot'] = None

    # Linux sample
    try:
        df_linux = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_Linux_dataset/train_test_linux_process.csv')
        X_linux, y_linux, _, _ = preprocess_data(df_linux, label_col='type', zero_day_attacks=[])

        normal_idx = np.where(y_linux == 0)[0][0]
        attack_idx = np.where(y_linux == 1)[0][0]

        samples['linux'] = {
            'normal': X_linux.iloc[normal_idx].values,
            'attack': X_linux.iloc[attack_idx].values
        }
        print("✅ Linux System samples loaded")
    except Exception as e:
        print(f"⚠️  Linux samples failed: {e}")
        samples['linux'] = None

    # Windows sample
    try:
        df_win = load_ton_iot_data('../data/Train_Test_datasets/Train_Test_Windows_dataset/train_test_windows_10.csv')
        X_win, y_win, _, _ = preprocess_data(df_win, label_col='type', zero_day_attacks=[])

        normal_idx = np.where(y_win == 0)[0][0]
        attack_idx = np.where(y_win == 1)[0][0]

        samples['windows'] = {
            'normal': X_win.iloc[normal_idx].values,
            'attack': X_win.iloc[attack_idx].values
        }
        print("✅ Windows System samples loaded")
    except Exception as e:
        print(f"⚠️  Windows samples failed: {e}")
        samples['windows'] = None

    print("="*60)
    return samples


def print_detection_result(scenario, results):
    """Pretty print detection results"""
    print(f"\n{results['fusion']['color']} {scenario}")
    print("="*60)

    # Layer results
    for layer_key in ['network', 'iot', 'linux', 'windows']:
        if layer_key in results and layer_key != 'fusion':
            r = results[layer_key]
            status = "🚨 ATTACK" if r['is_attack'] else "✅ Normal"
            print(f"  {layer_key.capitalize():15} {status:12} | Error: {r['error']:.6f} | Threshold: {r['threshold']:.6f}")
            if r['is_attack']:
                print(f"                  {'':12} | Severity: {r['severity']:.1f}%")

    # Fusion decision
    print("-"*60)
    fusion = results['fusion']
    print(f"  {'Fusion Score':15} {fusion['score']:.2f} (0.0-1.0)")
    print(f"  {'Threat Level':15} {fusion['color']} {fusion['threat_level']}")
    print(f"  {'Final Decision':15} {'🚨 ATTACK DETECTED!' if fusion['is_attack'] else '✅ SYSTEM NORMAL'}")
    print("="*60)


def demo_scenarios(ids, samples):
    """Run demo scenarios"""

    # Scenario 1: All Normal
    print("\n" + "="*70)
    print("📋 DEMO SCENARIO 1: NORMAL OPERATION")
    print("="*70)
    print("All layers receiving normal traffic...")

    normal_samples = {
        'network': samples['network']['normal'] if samples['network'] else None,
        'iot': samples['iot']['normal'] if samples['iot'] else None,
        'linux': samples['linux']['normal'] if samples['linux'] else None,
        'windows': samples['windows']['normal'] if samples['windows'] else None
    }
    results = ids.detect_multi_layer(normal_samples)
    print_detection_result("NORMAL TRAFFIC", results)

    # Scenario 2: Network Attack Only
    print("\n" + "="*70)
    print("📋 DEMO SCENARIO 2: NETWORK ATTACK")
    print("="*70)
    print("Suspicious network activity detected...")

    attack_samples_net = {
        'network': samples['network']['attack'] if samples['network'] else None,
        'iot': samples['iot']['normal'] if samples['iot'] else None,
        'linux': samples['linux']['normal'] if samples['linux'] else None,
        'windows': samples['windows']['normal'] if samples['windows'] else None
    }
    results = ids.detect_multi_layer(attack_samples_net)
    print_detection_result("SINGLE LAYER ATTACK", results)

    # Scenario 3: Multi-Layer Attack
    print("\n" + "="*70)
    print("📋 DEMO SCENARIO 3: COORDINATED MULTI-LAYER ATTACK")
    print("="*70)
    print("⚠️  Advanced Persistent Threat (APT) detected!")

    attack_samples_all = {
        'network': samples['network']['attack'] if samples['network'] else None,
        'iot': samples['iot']['attack'] if samples['iot'] else None,
        'linux': samples['linux']['attack'] if samples['linux'] else None,
        'windows': samples['windows']['normal'] if samples['windows'] else None
    }
    results = ids.detect_multi_layer(attack_samples_all)
    print_detection_result("MULTI-LAYER ATTACK", results)

    # Scenario 4: IoT-Specific Attack
    print("\n" + "="*70)
    print("📋 DEMO SCENARIO 4: IoT/SCADA ATTACK")
    print("="*70)
    print("Modbus protocol anomaly detected...")

    attack_samples_iot = {
        'network': samples['network']['normal'] if samples['network'] else None,
        'iot': samples['iot']['attack'] if samples['iot'] else None,
        'linux': samples['linux']['normal'] if samples['linux'] else None,
        'windows': samples['windows']['normal'] if samples['windows'] else None
    }
    results = ids.detect_multi_layer(attack_samples_iot)
    print_detection_result("IoT/SCADA ATTACK", results)


def main():
    """Main demo"""
    print("\n" + "="*70)
    print("🛡️  MINIMAL MULTI-LAYER IDS - INFERENCE DEMO")
    print("="*70)

    # Initialize IDS
    ids = MultiLayerIDS()

    # Load test samples
    samples = load_test_samples()

    # Run demo scenarios
    demo_scenarios(ids, samples)

    # Final summary
    print("\n" + "="*70)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📊 System Capabilities:")
    print("  ✅ 4-Layer detection (Network, IoT, Linux, Windows)")
    print("  ✅ Weighted fusion engine")
    print("  ✅ Real-time severity analysis")
    print("  ✅ Multi-layer threat correlation")
    print("\n💡 Next Steps:")
    print("  1. Deploy in Docker environment")
    print("  2. Connect to real traffic sources")
    print("  3. Set up Streamlit dashboard")
    print("  4. Configure alerting system")
    print("="*70)


if __name__ == '__main__':
    main()
