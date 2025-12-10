# 🌐 Real-time Zero-day Detection Deployment Guide

## Câu hỏi: "Trong môi trường thực tế, model nhận dữ liệu kiểu gì?"

---

## 📋 Tổng quan Pipeline Production

```
Internet Traffic → Capture → Extract Features → Scale → Model → Alert
```

---

## 1️⃣ TRAFFIC CAPTURE (Bắt gói tin)

### **Tools:**
- **Wireshark** / **tcpdump** - Packet capture
- **NetFlow** / **sFlow** - Flow-based monitoring
- **Zeek** (Bro) - Network security monitor
- **Suricata** - IDS/IPS with logging

### **Raw Data:**
```python
{
    'timestamp': '2025-12-08 10:30:45',
    'src_ip': '192.168.1.100',      # Source IP
    'dst_ip': '10.0.0.50',           # Destination IP
    'src_port': 47260,               # Source port
    'dst_port': 15600,               # Destination port
    'protocol': 'TCP',               # TCP/UDP/ICMP
    'bytes_sent': 1500,              # Payload size
    'bytes_received': 800,
    'duration': 0.5,                 # Connection duration
    'flags': 'SA',                   # TCP flags
    'ttl': 64,                       # Time to live
}
```

---

## 2️⃣ FEATURE EXTRACTION (Trích xuất đặc trưng)

### **Từ Raw Packet → 41 Features**

Model được train với 41 features, cần transform raw packet thành format này:

```python
def extract_features_from_packet(packet):
    """
    Convert raw network packet to 41 features
    """
    features = []

    # === NETWORK 5-TUPLE (5 features) ===
    features.append(packet['src_port'])        # Feature 1
    features.append(packet['dst_port'])        # Feature 2
    features.append(packet['duration'])        # Feature 3
    features.append(packet['bytes_sent'])      # Feature 4
    features.append(packet['bytes_received'])  # Feature 5

    # === CONNECTION STATS (10 features) ===
    features.append(packet.get('missed_bytes', 0))
    features.append(packet.get('retransmits', 0))
    features.append(packet.get('total_packets_sent', 0))
    features.append(packet.get('total_packets_received', 0))
    # ... 6 more connection features

    # === PROTOCOL FEATURES (One-hot encoding - 10 features) ===
    features.append(1 if packet['protocol'] == 'TCP' else 0)
    features.append(1 if packet['protocol'] == 'UDP' else 0)
    features.append(1 if packet['protocol'] == 'ICMP' else 0)
    # ... 7 more protocol types

    # === SERVICE FEATURES (10 features) ===
    features.append(1 if packet['dst_port'] == 80 else 0)   # HTTP
    features.append(1 if packet['dst_port'] == 443 else 0)  # HTTPS
    features.append(1 if packet['dst_port'] == 53 else 0)   # DNS
    # ... 7 more services

    # === TEMPORAL FEATURES (6 features) ===
    # Time-based patterns (hour of day, day of week, etc.)
    # Calculated from aggregated flows

    # Total: 41 features
    return np.array(features)
```

### **Aggregation Window:**
Thường aggregate traffic trong window (e.g., 5 seconds):
- Count packets
- Sum bytes
- Calculate ratios
- Detect patterns

---

## 3️⃣ PREPROCESSING (QUAN TRỌNG NHẤT!)

### **⚠️ MUST USE SAME SCALER FROM TRAINING!**

```python
import joblib

# Load scaler từ training
scaler = joblib.load('models/scaler.pkl')

# Apply transform lên RAW features
raw_features = extract_features_from_packet(packet)  # 41 features
scaled_features = scaler.transform(raw_features.reshape(1, -1))

# Bây giờ mới có thể đưa vào model!
```

### **Tại sao phải dùng SAME scaler?**

| | Training | Production |
|---|---|---|
| **src_port** | Mean=32,570, Std=22,216 | **MUST use same!** |
| **dst_port** | Mean=5,577, Std=13,034 | **MUST use same!** |
| **bytes** | Mean=1,234, Std=5,678 | **MUST use same!** |

**Nếu không dùng same scaler → Model nhận data sai scale → Error khổng lồ!**

---

## 4️⃣ MODEL INFERENCE

```python
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('models/ton_iot_autoencoder.h5', compile=False)
threshold = joblib.load('models/threshold.pkl')

# Predict
reconstructed = model.predict(scaled_features)
error = np.mean(np.power(scaled_features - reconstructed, 2))

# Detect
if error > threshold:
    alert("ATTACK DETECTED!", severity="CRITICAL")
else:
    log("Normal traffic")
```

---

## 5️⃣ ALERT & RESPONSE

```python
def handle_attack(packet, error, threshold):
    """
    Actions khi phát hiện attack
    """

    # 1. Calculate severity
    ratio = error / threshold
    if ratio < 1.5:
        severity = "LOW"
    elif ratio < 2.0:
        severity = "MEDIUM"
    elif ratio < 3.0:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    # 2. Block IP (firewall)
    os.system(f"iptables -A INPUT -s {packet['src_ip']} -j DROP")

    # 3. Alert SOC team
    send_email(
        to="soc@company.com",
        subject=f"[{severity}] Zero-day Attack Detected",
        body=f"Source: {packet['src_ip']}\\nError: {error:.6f}"
    )

    # 4. Log to SIEM
    syslog.warning(f"ATTACK | {packet['src_ip']} | {severity} | {error:.6f}")

    # 5. Capture full packet for forensics
    save_pcap(packet, f"attack_{packet['src_ip']}_{timestamp}.pcap")
```

---

## 6️⃣ DEPLOYMENT ARCHITECTURES

### **Option A: Inline IDS/IPS**
```
Internet → Firewall → [Detection System] → Internal Network
                           ↓
                      Block/Allow
```

### **Option B: Mirror Port (TAP)**
```
Internet → Switch (SPAN port) → Internal Network
              ↓
        [Detection System]
              ↓
        Alert Only (không block)
```

### **Option C: Agent-based**
```
Each IoT Device → Agent → Centralized Detection Server
```

---

## 7️⃣ EXAMPLE: Full Real-time Pipeline

```python
# ============================================
# COMPLETE REAL-TIME DETECTION SYSTEM
# ============================================

import pyshark  # Packet capture
import joblib
import numpy as np
from tensorflow import keras

class RealtimeZeroDayDetector:
    def __init__(self):
        # Load artifacts
        self.model = keras.models.load_model('models/ton_iot_autoencoder.h5', compile=False)
        self.scaler = joblib.load('models/scaler.pkl')
        self.threshold = joblib.load('models/threshold.pkl')

    def capture_traffic(self, interface='eth0'):
        """
        Capture live traffic from network interface
        """
        capture = pyshark.LiveCapture(interface=interface)

        for packet in capture.sniff_continuously():
            # Extract features
            features = self.extract_features(packet)

            # Detect
            is_attack = self.detect(features)

            if is_attack:
                self.alert(packet)

    def extract_features(self, packet):
        """
        Parse packet → 41 features
        """
        try:
            features = [
                int(packet.tcp.srcport) if hasattr(packet, 'tcp') else 0,
                int(packet.tcp.dstport) if hasattr(packet, 'tcp') else 0,
                float(packet.sniff_time.timestamp()),
                int(packet.length),
                # ... extract 37 more features
            ]

            # Pad to 41
            while len(features) < 41:
                features.append(0.0)

            return np.array(features[:41])

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def detect(self, raw_features):
        """
        Detect attack from raw features
        """
        if raw_features is None:
            return False

        # CRITICAL: Scale with training scaler!
        scaled = self.scaler.transform(raw_features.reshape(1, -1))

        # Predict
        reconstructed = self.model.predict(scaled, verbose=0)
        error = np.mean(np.power(scaled - reconstructed, 2))

        return error > self.threshold

    def alert(self, packet):
        """
        Handle attack detection
        """
        print(f"🚨 ATTACK DETECTED from {packet.ip.src}")

        # Block IP
        # os.system(f"iptables -A INPUT -s {packet.ip.src} -j DROP")

        # Log to file
        with open('attacks.log', 'a') as f:
            f.write(f"{packet.sniff_time} | {packet.ip.src} | ATTACK\\n")

# Run detector
detector = RealtimeZeroDayDetector()
detector.capture_traffic(interface='eth0')
```

---

## 8️⃣ PERFORMANCE OPTIMIZATION

### **Latency Requirements:**
- Packet processing: < 10ms
- Feature extraction: < 5ms
- Model inference: < 2ms
- Total: < 20ms per packet

### **Optimizations:**
1. **Batch processing** - Process multiple packets together
2. **GPU acceleration** - Use CUDA for inference
3. **Model quantization** - FP16 instead of FP32
4. **Feature caching** - Cache common patterns
5. **Sampling** - Check 10% of traffic initially

---

## 9️⃣ MONITORING & MAINTENANCE

### **Metrics to Track:**
- Throughput: packets/second
- Latency: ms per detection
- False Positive Rate: %
- False Negative Rate: %
- Model drift: error distribution over time

### **Model Retraining:**
```
Every 30 days OR when:
- False positive rate > 10%
- New attack types emerge
- Network topology changes
```

---

## 🎯 TÓM TẮT: Production vs Lab

| Aspect | Lab (Current) | Production (Real) |
|--------|---------------|-------------------|
| **Input** | Scaled numpy array | Raw network packets |
| **Features** | 41 pre-calculated | Extract from packets |
| **Preprocessing** | Already done | **MUST apply scaler** |
| **Latency** | No requirement | < 20ms |
| **Scale** | 106K test samples | Millions packets/day |
| **Actions** | Print result | Block IP, Alert SOC |

---

## 📚 Tools & Libraries

### **Packet Capture:**
```bash
pip install pyshark scapy
```

### **Real-time Processing:**
```bash
pip install apache-kafka redis celery
```

### **Model Serving:**
```bash
pip install tensorflow-serving fastapi uvicorn
```

### **Monitoring:**
```bash
pip install prometheus-client grafana-api
```

---

## 🚀 Next Steps

1. ✅ Model đã train tốt (95% accuracy)
2. ✅ Test với scaled data thành công
3. 🔄 **Cần làm:** Integrate với packet capture system
4. 🔄 **Cần làm:** Deploy lên production server
5. 🔄 **Cần làm:** Setup monitoring & alerting

**KEY POINT:** Trong production, phải có **Feature Extraction Pipeline** để convert raw packets → 41 features → scale với SAME scaler → model inference!
