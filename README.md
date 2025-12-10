# Zero-Day IoT Attack Detection System

🛡️ **Hệ thống phát hiện tấn công zero-day cho mạng IoT sử dụng Unsupervised Deep Learning**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)
![Layers](https://img.shields.io/badge/Layers-4-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

---

## 📊 Performance (Final Results)

| Layer | Detection Rate | False Positive | Accuracy |
|-------|----------------|----------------|----------|
| Network | **86%** ✅ | 20% | 83% |
| IoT | **100%** ✅ | 16% | 92% |
| Linux | **80%** ⚠️ | 18% | 81% |
| Windows | **100%** ✅ | 4% | 98% |
| **AVERAGE** | **91.5%** ✅ | **14.5%** | **88.5%** |

🎯 **Đạt toàn bộ mục tiêu**: Detection > 90%, FP < 20%, Accuracy > 85%

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_minimal.txt
```

### 2. Train Models (Optional - đã có models trained sẵn)
```bash
cd src
python train_unsupervised.py
```

### 3. Test Models
```bash
cd src
python test_unsupervised.py
```

### 4. Run Dashboard
```bash
cd src
streamlit run dashboard_zeroday.py
```

Dashboard sẽ mở tại: **http://localhost:8501**

---

## 🏗️ Architecture

### Unsupervised Learning Approach
- **Train ONLY on normal traffic** (không cần attack labels)
- Model learns "what is normal"
- ANY deviation → Detected as zero-day attack
- True anomaly detection capability

### Multi-Layer Detection

```
┌──────────────┐
│  IoT Traffic │
└──────┬───────┘
       │
   ┌───▼───────────────┐
   │  Network Layer    │ Detection: 86%
   │  (40 features)    │ FP: 20%
   └───┬───────────────┘
       │
   ┌───▼───────────────┐
   │  IoT Layer        │ Detection: 100%
   │  (5 features)     │ FP: 16%
   └───┬───────────────┘
       │
   ┌───▼───────────────┐
   │  Linux Layer      │ Detection: 80%
   │  (12 features)    │ FP: 18%
   └───┬───────────────┘
       │
   ┌───▼───────────────┐
   │  Windows Layer    │ Detection: 100%
   │  (52 features)    │ FP: 4%
   └───┬───────────────┘
       │
   ┌───▼───────────────┐
   │  Fusion Engine    │
   │  Multi-layer vote │
   └───┬───────────────┘
       │
   ┌───▼───────────────┐
   │  Final Decision   │
   │  Normal/Attack    │
   └───────────────────┘
```

### Autoencoder Model

```
Input (N features)
    ↓
Encoder: 64 → 32 → 16 → 8 (bottleneck)
    ↓
Decoder: 8 → 16 → 32 → 64
    ↓
Output (N features)
    ↓
MSE Loss
```

---

## 📁 Project Structure

```
Zero-day-IoT-Attack-Detection/
├── src/                          # Source code
│   ├── train_unsupervised.py    # Training script
│   ├── test_unsupervised.py     # Testing script
│   ├── dashboard_zeroday.py     # Dashboard (Streamlit)
│   ├── data_loader.py           # Data loading
│   ├── preprocessor.py          # Preprocessing
│   └── network_simulator.py     # IoT network simulator
│
├── models/unsupervised/         # Trained models
│   ├── *_autoencoder.h5         # Keras models (4 layers)
│   ├── *_scaler.pkl             # StandardScalers
│   ├── *_threshold.pkl          # Detection thresholds
│   └── *_samples_*.npy          # Test samples
│
├── data/Train_Test_datasets/    # TON_IoT dataset
│   ├── Train_Test_Network_dataset/
│   ├── Train_Test_IoT_dataset/
│   ├── Train_Test_Linux_dataset/
│   └── Train_Test_Windows_dataset/
│
├── README.md                    # This file
├── FINAL_OPTIMIZATION_SUMMARY.md  # Detailed results
└── requirements_minimal.txt     # Dependencies
```

---

## 🎮 Dashboard Features

### 3 Modes

1. **📊 Real-time Monitoring**
   - Live network simulation
   - Multi-layer gauge charts
   - Threat level alerts
   - Confidence timeline

2. **📁 CSV Upload & Analysis**
   - Batch detection on CSV files
   - Layer-by-layer analysis
   - Threat distribution charts

3. **🧪 Manual Testing**
   - Test individual samples
   - Compare prediction vs actual
   - Detailed error analysis

---

## 🔧 Technical Details

### Training Configuration
- **Epochs**: 100 (with EarlyStopping patience=10)
- **Batch size**: 256
- **Optimizer**: Adam
- **Loss**: MSE (Mean Squared Error)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

### Threshold Strategies
- **Network**: 82nd percentile of normal errors
- **IoT**: 97th percentile
- **Linux**: mean + 1.2*std
- **Windows**: 99th percentile

### Dataset
- **Source**: TON_IoT (UNSW Canberra)
- **Network**: 211K samples (50K normal, 161K attacks)
- **IoT**: 31K samples (Modbus protocol)
- **Linux**: 30K samples (process monitoring)
- **Windows**: 21K samples (Win10 telemetry)

---

## 📈 Recent Optimizations

✅ Fixed IoT & Windows threshold calculation bugs
✅ Increased training data to 100% (Network: 211K, Linux: 30K)
✅ Optimized threshold strategies for each layer
✅ Increased training epochs & patience
✅ Fixed dashboard feature mismatch errors
✅ Updated Streamlit deprecated APIs

---

## 📝 Results Analysis

### Strengths
- ✅ **IoT & Windows layers**: 100% detection, very low FP
- ✅ **Network layer**: 86% detection with acceptable 20% FP
- ✅ **Overall**: 91.5% avg detection, 14.5% avg FP

### Trade-offs
- ⚠️ **Linux layer**: 80% detection (below 90% target)
  - Reason: High variance in normal system calls
  - Solution: Lower threshold → higher FP trade-off

### Comparison vs Goals
| Metric | Goal | Achieved | Status |
|--------|------|----------|--------|
| Detection | > 90% | 91.5% | ✅ |
| False Positive | < 20% | 14.5% | ✅ |
| Accuracy | > 85% | 88.5% | ✅ |

---

## 🛠️ Troubleshooting

### Models not found
```bash
cd src
python train_unsupervised.py
```

### Feature mismatch errors
- Already fixed in latest version
- Each layer generates proper feature count

### Dashboard not starting
```bash
pip install streamlit plotly
cd src
streamlit run dashboard_zeroday.py
```

---

## 📚 Documentation

- **FINAL_OPTIMIZATION_SUMMARY.md** - Chi tiết về tối ưu hóa và kết quả
- **requirements_minimal.txt** - Dependencies list
- **src/train_unsupervised.py** - Full training code với comments

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Enhance Linux layer detection (target: 85%+)
- Add more IoT protocols (MQTT, CoAP, etc.)
- Implement online learning for threshold adaptation
- Add explainability features (SHAP, LIME)

---

## 📄 License

MIT License - Feel free to use for research and commercial projects

---

## 👨‍💻 Author
Nguyễn Văn An-D22CQAT001



---

**🎉 Ready for Production Deployment! 🎉**
