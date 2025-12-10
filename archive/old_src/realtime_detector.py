"""
Real-time Zero-day Attack Detection System
Simulates production deployment
"""

import numpy as np
import pandas as pd
from inference import ZeroDayDetector
import time


class RealtimeNetworkMonitor:
    """
    Real-time network traffic monitor with zero-day detection
    """

    def __init__(self, detector):
        self.detector = detector
        self.alert_count = 0

    def extract_features_from_packet(self, packet_data):
        """
        BƯỚC 1: Trích xuất features từ raw network packet

        Input: packet_data (dict) - Raw packet từ network capture
        Output: 41 features (numpy array)

        Trong thực tế, đây là nơi bạn parse:
        - IP addresses → Encode (one-hot, hash, hoặc numeric)
        - Ports → Numeric
        - Protocol → Encode (TCP=6, UDP=17, etc.)
        - Bytes, packets, duration → Aggregate từ flow
        """

        # Giả lập feature extraction từ packet
        features = []

        # Network 5-tuple
        features.append(packet_data.get('src_port', 0))
        features.append(packet_data.get('dst_port', 0))
        features.append(packet_data.get('duration', 0))
        features.append(packet_data.get('src_bytes', 0))
        features.append(packet_data.get('dst_bytes', 0))

        # Connection statistics (aggregated over time window)
        features.extend([
            packet_data.get('missed_bytes', 0),
            packet_data.get('src_pkts', 0),
            packet_data.get('src_ip_bytes', 0),
            packet_data.get('dst_pkts', 0),
            packet_data.get('dst_ip_bytes', 0),
        ])

        # Protocol features
        features.extend([
            packet_data.get('proto_tcp', 0),
            packet_data.get('proto_udp', 0),
            packet_data.get('proto_icmp', 0),
            # ... more protocol features
        ])

        # Temporal features (time-based patterns)
        features.extend([
            packet_data.get('conn_state', 0),
            packet_data.get('history', 0),
            # ... more temporal features
        ])

        # Service features
        features.extend([
            packet_data.get('service_http', 0),
            packet_data.get('service_dns', 0),
            packet_data.get('service_ssl', 0),
            # ... more service features
        ])

        # Pad/truncate to 41 features
        while len(features) < 41:
            features.append(0.0)
        features = features[:41]

        return np.array(features)

    def process_traffic(self, raw_packet):
        """
        BƯỚC 2-5: Xử lý traffic và detect attack

        Input: raw_packet (dict) - Raw network packet
        Output: Detection result
        """

        # Extract features
        raw_features = self.extract_features_from_packet(raw_packet)

        # Detect (scaler.transform() được gọi bên trong với already_scaled=False)
        result = self.detector.detect(
            raw_features,
            return_details=True,
            already_scaled=False  # Raw features → sẽ được scale
        )

        # Handle alert
        if result['prediction'] == 'ATTACK':
            self.alert_count += 1
            self.handle_attack(raw_packet, result)

        return result

    def handle_attack(self, packet, result):
        """
        BƯỚC 6: Xử lý khi phát hiện attack
        """
        print(f"\n{'='*60}")
        print(f"🚨 ATTACK DETECTED - Alert #{self.alert_count}")
        print(f"{'='*60}")
        print(f"Source: {packet.get('src_ip', 'unknown')}:{packet.get('src_port', 0)}")
        print(f"Destination: {packet.get('dst_ip', 'unknown')}:{packet.get('dst_port', 0)}")
        print(f"Severity: {result['severity']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Reconstruction Error: {result['reconstruction_error']:.6f}")
        print(f"Threshold: {result['threshold']:.6f}")

        # TODO: Thực hiện actions
        # - Block IP in firewall
        # - Send alert to SOC
        # - Log to SIEM
        # - Trigger incident response

        print(f"\n[ACTION] Blocking IP {packet.get('src_ip', 'unknown')}")
        print(f"[ACTION] Alerting SOC team")
        print(f"[ACTION] Logging to SIEM")


def simulate_realtime_traffic():
    """
    Demo: Giả lập real-time traffic monitoring
    """

    print("="*60)
    print("REAL-TIME ZERO-DAY DETECTION SYSTEM")
    print("="*60)
    print("\nInitializing detector...")

    # Load detector
    detector = ZeroDayDetector(
        model_path="../models/ton_iot_autoencoder.h5",
        scaler_path="../models/scaler.pkl",
        threshold_path="../models/threshold.pkl",
    )

    monitor = RealtimeNetworkMonitor(detector)

    print("✓ Detector initialized")
    print("✓ Monitoring started...\n")

    # Giả lập traffic stream
    traffic_samples = [
        # Normal IoT traffic
        {
            'src_ip': '192.168.1.100',
            'dst_ip': '10.0.0.50',
            'src_port': 47260,
            'dst_port': 15600,
            'duration': 0.1,
            'src_bytes': 100,
            'dst_bytes': 50,
            'src_pkts': 1,
            'dst_pkts': 1,
            'missed_bytes': 0,
            'src_ip_bytes': 150,
            'dst_ip_bytes': 80,
            'proto_tcp': 1,
            'proto_udp': 0,
            'conn_state': 1,
            'service_http': 1,
        },

        # Normal DNS query
        {
            'src_ip': '192.168.1.101',
            'dst_ip': '8.8.8.8',
            'src_port': 52341,
            'dst_port': 53,
            'duration': 0.05,
            'src_bytes': 64,
            'dst_bytes': 128,
            'src_pkts': 1,
            'dst_pkts': 1,
            'missed_bytes': 0,
            'src_ip_bytes': 92,
            'dst_ip_bytes': 156,
            'proto_udp': 1,
            'service_dns': 1,
        },

        # ATTACK: DDoS pattern (high volume, suspicious port)
        {
            'src_ip': '203.0.113.66',  # External attacker
            'dst_ip': '192.168.1.10',
            'src_port': 4444,  # Suspicious
            'dst_port': 80,
            'duration': 120.5,  # Very long
            'src_bytes': 250000,  # Very high!
            'dst_bytes': 5000,
            'src_pkts': 500,  # Flood!
            'dst_pkts': 50,
            'missed_bytes': 100,
            'src_ip_bytes': 250500,
            'dst_ip_bytes': 5100,
            'proto_tcp': 1,
            'conn_state': 5,  # Anomalous
            'service_http': 1,
        },

        # ATTACK: Port scan
        {
            'src_ip': '198.51.100.23',
            'dst_ip': '192.168.1.50',
            'src_port': 54321,
            'dst_port': 22,  # SSH
            'duration': 0.01,  # Very fast
            'src_bytes': 40,  # Small SYN packet
            'dst_bytes': 0,  # No response
            'src_pkts': 1,
            'dst_pkts': 0,
            'missed_bytes': 0,
            'src_ip_bytes': 60,
            'dst_ip_bytes': 0,
            'proto_tcp': 1,
            'conn_state': 0,  # Connection failed
        },
    ]

    # Process each packet
    for i, packet in enumerate(traffic_samples):
        print(f"\n[{i+1}] Processing packet from {packet['src_ip']}:{packet['src_port']} → {packet['dst_ip']}:{packet['dst_port']}")

        result = monitor.process_traffic(packet)

        if result['prediction'] == 'NORMAL':
            print(f"    ✓ NORMAL traffic (error: {result['reconstruction_error']:.6f})")

        # Simulate real-time delay
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"MONITORING SUMMARY")
    print(f"{'='*60}")
    print(f"Total packets processed: {len(traffic_samples)}")
    print(f"Attacks detected: {monitor.alert_count}")
    print(f"Normal traffic: {len(traffic_samples) - monitor.alert_count}")


if __name__ == "__main__":
    simulate_realtime_traffic()
