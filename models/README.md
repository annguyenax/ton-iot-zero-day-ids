# Models Directory

This directory contains trained models and preprocessing artifacts.

## 📁 Directory Structure

```
models/
└── unsupervised/              # Unsupervised autoencoder models
    ├── network_*.{h5,pkl,npy}    # Network layer (4 files)
    ├── iot_*.{h5,pkl,npy}        # IoT layer (4 files)
    ├── linux_*.{h5,pkl,npy}      # Linux layer (4 files)
    └── windows_*.{h5,pkl,npy}    # Windows layer (4 files)
```

## 📦 Files Generated Per Layer (8 files each)

After running `python src/train_unsupervised.py`, each layer generates:

### Core Model Files (Existed before)

| File | Size Example | Purpose |
|------|-------------|---------|
| `*_autoencoder.h5` | ~150-200K | Keras model (weights + architecture) |
| `*_scaler.pkl` | ~1-2K | StandardScaler (mean, std for normalization) |
| `*_threshold.pkl` | 117 bytes | Optimal anomaly detection threshold |
| `*_samples_X.npy` | ~10-40K | Test samples (50 normal + 50 attack) |
| `*_samples_y.npy` | 928 bytes | Test labels (0=normal, 1=attack) |

### Reproducibility Files (NEW! ⭐)

| File | Size Example | Purpose |
|------|-------------|---------|
| `*_encoders.pkl` | ~500B-300K | LabelEncoders for categorical features (sorted) |
| `*_feature_names.pkl` | ~100B-2K | Feature column names in order |
| `*_metadata.pkl` | ~500B-2K | Training info (metrics, config, params) |

**New files ensure 100% reproducibility and proper inference!**

## 🔍 File Details

### 1. `{layer}_autoencoder.h5`

**Keras model file containing:**
- Model architecture (encoder-decoder)
- Trained weights
- Optimizer state

**Size by layer:**
- Network: ~197K (40 features → 64 → 32 → 16 → 8 → 16 → 32 → 64 → 40)
- IoT: ~143K (5 features)
- Linux: ~153K (12 features)
- Windows: ~215K (52 features)

### 2. `{layer}_scaler.pkl`

**StandardScaler object:**
```python
scaler = StandardScaler()
scaler.fit(X_train)  # Fitted on NORMAL training data only
```

Contains:
- `mean_`: Mean of each feature
- `scale_`: Standard deviation of each feature

**Usage:**
```python
scaler = joblib.load('network_scaler.pkl')
X_scaled = scaler.transform(X_new)
```

### 3. `{layer}_threshold.pkl`

**Single float value:**
- Optimal reconstruction error threshold
- Calculated from normal training errors
- Layer-specific (different strategies)

**Example values:**
- Network: 0.117323 (82nd percentile)
- IoT: 0.142731 (97th percentile)
- Linux: 0.058823 (mean + 1.2*std)
- Windows: 0.688233 (99th percentile)

**Usage:**
```python
threshold = joblib.load('network_threshold.pkl')
is_attack = (reconstruction_error > threshold)
```

### 4. `{layer}_samples_X.npy` & `{layer}_samples_y.npy`

**Test samples for quick validation:**
- 50 normal samples
- 50 attack samples
- Used by dashboard and test scripts

**Usage:**
```python
X_test = np.load('network_samples_X.npy')  # (100, n_features)
y_test = np.load('network_samples_y.npy')  # (100,)
```

### 5. `{layer}_encoders.pkl` ⭐ NEW!

**Dictionary of LabelEncoders:**
```python
{
    'proto': LabelEncoder(['icmp', 'tcp', 'udp']),  # Sorted alphabetically!
    'service': LabelEncoder(['dns', 'ftp', 'http', ...]),
    'conn_state': LabelEncoder(['CON', 'FIN', 'INT', ...]),
    ...
}
```

**Why important:**
- ✅ Ensures deterministic encoding (sorted)
- ✅ Required for inference on new data
- ✅ Prevents feature mismatch errors

**Size:**
- Network: 295K (many categorical: IPs, protocols, services)
- IoT: 560 bytes (few categorical)
- Linux: 665 bytes (few categorical)
- Windows: 1.4K (few categorical)

**Usage:**
```python
encoders = joblib.load('network_encoders.pkl')
for col in categorical_cols:
    X[col] = encoders[col].transform(X[col])
```

### 6. `{layer}_feature_names.pkl` ⭐ NEW!

**List of feature names in order:**
```python
['src_port', 'dst_port', 'proto', 'service', 'duration', ...]
```

**Why important:**
- ✅ Ensures correct feature order during inference
- ✅ Helps debugging (know which feature causes issues)
- ✅ Documentation (know what features are used)

**Usage:**
```python
feature_names = joblib.load('network_feature_names.pkl')
print(f"Model expects {len(feature_names)} features: {feature_names}")
```

### 7. `{layer}_metadata.pkl` ⭐ NEW!

**Dictionary with training information:**
```python
{
    'layer_name': 'network',
    'n_features': 40,
    'feature_names': ['src_port', ...],
    'categorical_features': ['proto', 'service', ...],
    'threshold': 0.117323,
    'threshold_method': '82nd percentile',
    'n_train': 35000,
    'n_val': 7500,
    'n_test_normal': 7500,
    'n_test_attack': 1000,
    'false_positive_rate': 0.2652,
    'detection_rate': 0.7760,
    'separation_ratio': 328.17,
}
```

**Why important:**
- ✅ Full audit trail of training
- ✅ Compare different training runs
- ✅ Reproduce exact training config
- ✅ Documentation & transparency

**Usage:**
```python
metadata = joblib.load('network_metadata.pkl')
print(f"Model trained on {metadata['n_train']} samples")
print(f"Detection rate: {metadata['detection_rate']:.2%}")
```

---

## 📊 Total Size

**Per layer:** ~300-400K (8 files)
**All layers:** ~1.2 MB (32 files)

This is very small and acceptable for version control, **BUT** we follow ML best practice of not committing trained models to Git.

---

## ⚠️ Important Notes

### Models are NOT in Git Repository

**Why?**
1. ML best practice: Users train their own models
2. Ensures reproducibility: Users verify results
3. Different users may need different configs

**How to get models:**
```bash
# After cloning the repo
cd src
python train_unsupervised.py
# This generates all 32 files in models/unsupervised/
```

### Reproducibility Guarantee

With the new code (sorted encoding + fixed seeds):
- ✅ Training is **100% reproducible**
- ✅ Same results every time on any machine
- ✅ Detection rates, FP rates, thresholds are identical

**Verify yourself:**
```bash
# Train twice
python train_unsupervised.py > log1.txt
python train_unsupervised.py > log2.txt

# Compare - should be identical
diff log1.txt log2.txt  # No differences!
```

---

## 🔄 Retraining Models

**When to retrain:**
1. New data available
2. Changed hyperparameters (epochs, batch size, etc.)
3. Modified preprocessing logic
4. Different threshold strategy
5. Deploy to new environment (verify performance)

**How to retrain:**
```bash
# Backup old models (optional)
cd models
mv unsupervised unsupervised_backup_$(date +%Y%m%d)
mkdir unsupervised

# Train new models
cd ../src
python train_unsupervised.py
```

**Training time:**
- CPU: ~40-60 minutes
- GPU: ~10-15 minutes

---

## 📚 Model Architecture

Each layer uses the same autoencoder architecture:

```
Input (N features)
    ↓
Encoder:
  Dense(64) + ReLU + Dropout(0.2)
  Dense(32) + ReLU + Dropout(0.2)
  Dense(16) + ReLU + Dropout(0.2)
  Dense(8)  + ReLU + Dropout(0.2)  ← Bottleneck
    ↓
Decoder:
  Dense(16) + ReLU + Dropout(0.2)
  Dense(32) + ReLU + Dropout(0.2)
  Dense(64) + ReLU + Dropout(0.2)
  Dense(N)  + Linear                ← Reconstruction
    ↓
MSE(input, reconstruction) > threshold? → ATTACK : NORMAL
```

**Training config:**
- Optimizer: Adam
- Loss: MSE (Mean Squared Error)
- Epochs: 100 (with EarlyStopping patience=10)
- Batch size: 256
- Trained on: **NORMAL traffic only** (unsupervised)

---

## 🎯 Performance Summary

| Layer | Detection | FP | Threshold | Separation |
|-------|-----------|----|-----------| -----------|
| **Network** | 77.6% | 26.5% | 0.117323 | 328x |
| **IoT** | 100% | 7.4% | 0.142731 | 33x |
| **Linux** | 79.6% | 24.0% | 0.058823 | 8x |
| **Windows** | 100% | 6.9% | 0.688233 | 36x |
| **AVERAGE** | **89.3%** | **16.2%** | - | **101x** |

🎯 **100% Reproducible - Same results every time!**

---

## 📖 Loading Models (Example)

```python
import numpy as np
import joblib
from tensorflow import keras

# Load all artifacts for a layer
model = keras.models.load_model('models/unsupervised/network_autoencoder.h5')
scaler = joblib.load('models/unsupervised/network_scaler.pkl')
threshold = joblib.load('models/unsupervised/network_threshold.pkl')
encoders = joblib.load('models/unsupervised/network_encoders.pkl')
feature_names = joblib.load('models/unsupervised/network_feature_names.pkl')
metadata = joblib.load('models/unsupervised/network_metadata.pkl')

# Preprocess new data
X_new = preprocess_data(df)  # Returns DataFrame
for col in encoders.keys():
    X_new[col] = encoders[col].transform(X_new[col])

# Scale
X_scaled = scaler.transform(X_new)

# Predict
X_reconstructed = model.predict(X_scaled)
errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

# Detect anomalies
predictions = (errors > threshold).astype(int)
print(f"Detected {predictions.sum()} anomalies out of {len(predictions)} samples")
```

---

## 🔗 See Also

- [Training script](../src/train_unsupervised.py) - How models are trained
- [Testing script](../src/test_unsupervised.py) - How to test models
- [Preprocessor](../src/preprocessor.py) - Feature preprocessing logic
- [README](../README.md) - Main project documentation
