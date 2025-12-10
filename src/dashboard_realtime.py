"""
Multi-Layer IDS - Real-time Monitoring Dashboard
Simulates live traffic with attack detection alerts
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from tensorflow import keras
import warnings
import time
import datetime
from io import StringIO

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Multi-Layer IDS - Real-time Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .status-normal {
        background-color: #28a745;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-attack {
        background-color: #dc3545;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        animation: glow-red 2s ease-in-out infinite;
        box-shadow: 0 0 20px rgba(220,53,69,0.6);
    }
    @keyframes glow-red {
        0%, 100% { box-shadow: 0 0 20px rgba(220,53,69,0.6); }
        50% { box-shadow: 0 0 30px rgba(220,53,69,0.9); }
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .traffic-log {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

class MultiLayerIDS:
    """Multi-Layer IDS with real-time detection"""

    def __init__(self, models_dir='../models/minimal'):
        self.models_dir = models_dir
        self.layers = {}

        # Load all 4 layers
        layer_configs = [
            ('network', 'Network Traffic', 0.30),
            ('iot', 'IoT/SCADA Modbus', 0.25),
            ('linux', 'Linux System', 0.25),
            ('windows', 'Windows System', 0.20)
        ]

        for key, name, weight in layer_configs:
            self.layers[key] = {
                'name': name,
                'model': keras.models.load_model(f'{models_dir}/{key}_autoencoder.h5', compile=False),
                'scaler': joblib.load(f'{models_dir}/{key}_scaler.pkl'),
                'threshold': joblib.load(f'{models_dir}/{key}_threshold.pkl'),
                'weight': weight
            }

    def detect_layer(self, layer_key, features):
        """Detect anomaly in single layer"""
        layer = self.layers[layer_key]

        scaled = layer['scaler'].transform(features.reshape(1, -1))
        reconstructed = layer['model'].predict(scaled, verbose=0)
        error = np.mean(np.power(scaled - reconstructed, 2))

        is_attack = error > layer['threshold']
        severity = min(100, (error / layer['threshold'] - 1) * 100) if is_attack else 0

        return {
            'is_attack': is_attack,
            'error': error,
            'threshold': layer['threshold'],
            'severity': severity
        }

    def fusion_decision(self, detections):
        """Fusion engine - weighted voting"""
        fusion_score = sum(
            self.layers[k]['weight']
            for k, r in detections.items()
            if r['is_attack']
        )

        if fusion_score >= 0.5:
            threat_level, color = 'CRITICAL', 'red'
        elif fusion_score >= 0.3:
            threat_level, color = 'HIGH', 'orange'
        elif fusion_score >= 0.1:
            threat_level, color = 'MEDIUM', 'yellow'
        else:
            threat_level, color = 'LOW', 'green'

        return {
            'score': fusion_score,
            'threat_level': threat_level,
            'color': color,
            'is_attack': fusion_score >= 0.3
        }

def create_synthetic_attack(normal_sample, noise_level=2.0):
    """Create synthetic attack by adding Gaussian noise"""
    noise = np.random.randn(*normal_sample.shape) * noise_level * np.std(normal_sample)
    return normal_sample + noise

def load_csv_data(uploaded_file, layer_key, ids):
    """Load and process CSV file for specific layer"""
    try:
        df = pd.read_csv(uploaded_file)

        st.info(f"📊 CSV loaded: {len(df)} rows, {len(df.columns)} columns")

        # Try to detect if it has a label column
        label_col = None
        for col in ['label', 'Label', 'attack', 'Attack', 'class', 'Class', 'type', 'Type']:
            if col in df.columns:
                label_col = col
                break

        # Remove label column if exists
        if label_col:
            labels = df[label_col].values
            # Convert to binary if needed
            if labels.dtype == object:
                # Map normal/attack to 0/1
                labels = np.array([0 if str(v).lower() in ['normal', '0', 'benign'] else 1 for v in labels])
            df = df.drop(columns=[label_col])
            st.success(f"✅ Detected label column: '{label_col}'")
        else:
            labels = None
            st.warning("⚠️ No label column detected. All samples will be analyzed.")

        # Drop non-numeric columns (timestamp, date, categorical)
        non_numeric_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
        if non_numeric_cols:
            st.warning(f"⚠️ Dropping non-numeric columns: {', '.join(non_numeric_cols)}")
            df = df.drop(columns=non_numeric_cols)

        # Drop any columns with all NaN
        df = df.dropna(axis=1, how='all')

        # Fill remaining NaN with 0
        df = df.fillna(0)

        # Convert to numeric, coerce errors to NaN then fill with 0
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)

        # Check if we have any features left
        if df.shape[1] == 0:
            st.error("❌ No numeric features found in CSV!")
            return None, None

        st.success(f"✅ Processed: {len(df)} samples, {df.shape[1]} numeric features")

        # Convert to numpy array
        features = df.values

        return features, labels
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

@st.cache_resource
def load_ids():
    """Load IDS model"""
    return MultiLayerIDS()

@st.cache_data
def load_samples():
    """Load test samples"""
    samples = {}
    for layer in ['network', 'iot', 'linux', 'windows']:
        X = np.load(f'../models/minimal/{layer}_samples_X.npy')
        y = np.load(f'../models/minimal/{layer}_samples_y.npy')
        samples[layer] = {'X': X, 'y': y}
    return samples

def plot_mini_gauge(value, threshold, title, is_attack):
    """Create mini gauge chart"""
    color = "red" if is_attack else "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, threshold * 3]},
            'bar': {'color': color},
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))

    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Multi-Layer IDS - Real-time Monitoring</div>',
                unsafe_allow_html=True)

    # Load models and samples
    ids = load_ids()
    samples = load_samples()

    # Initialize session state
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'traffic_log' not in st.session_state:
        st.session_state.traffic_log = []
    if 'packet_count' not in st.session_state:
        st.session_state.packet_count = 0
    if 'attack_count' not in st.session_state:
        st.session_state.attack_count = 0

    # Sidebar
    st.sidebar.title("⚙️ Control Panel")

    mode = st.sidebar.radio(
        "Mode",
        ["🎬 Live Simulation", "📁 Upload CSV", "🎯 Manual Test"]
    )

    st.sidebar.markdown("---")

    if mode == "🎬 Live Simulation":
        st.sidebar.subheader("Simulation Settings")

        traffic_rate = st.sidebar.slider("Traffic Rate (packets/sec)", 1, 10, 2)
        attack_prob = st.sidebar.slider("Attack Probability (%)", 0, 100, 20)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.monitoring = True
        with col2:
            if st.button("⏸️ Stop", use_container_width=True):
                st.session_state.monitoring = False

        if st.sidebar.button("🔄 Reset Stats", use_container_width=True):
            st.session_state.detection_history = []
            st.session_state.traffic_log = []
            st.session_state.packet_count = 0
            st.session_state.attack_count = 0

    elif mode == "📁 Upload CSV":
        st.sidebar.subheader("Upload Data")

        selected_layer = st.sidebar.selectbox(
            "Layer",
            ["network", "iot", "linux", "windows"],
            format_func=lambda x: ids.layers[x]['name']
        )

        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV file with features (one row per sample)"
        )

        if uploaded_file and st.sidebar.button("🔍 Analyze", use_container_width=True):
            features, labels = load_csv_data(uploaded_file, selected_layer, ids)

            if features is not None:
                st.session_state.csv_results = {
                    'layer': selected_layer,
                    'features': features,
                    'labels': labels
                }

    else:  # Manual Test
        st.sidebar.subheader("Manual Test")

        scenario = st.sidebar.selectbox(
            "Scenario",
            ["Normal Traffic", "Single Layer Attack", "Multi-Layer Attack"]
        )

        if st.sidebar.button("🔍 Run Test", use_container_width=True):
            st.session_state.manual_test = scenario

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **System Info:**
    - 4-Layer Detection
    - Real-time Analysis
    - Fusion Engine
    - Alert System
    """)

    # Main content area
    if mode == "🎬 Live Simulation":
        # Status display
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📦 Packets Analyzed", st.session_state.packet_count)
        with col2:
            st.metric("🚨 Attacks Detected", st.session_state.attack_count)
        with col3:
            detection_rate = (st.session_state.attack_count / max(st.session_state.packet_count, 1)) * 100
            st.metric("📊 Detection Rate", f"{detection_rate:.1f}%")
        with col4:
            if st.session_state.detection_history:
                avg_score = np.mean([h['fusion_score'] for h in st.session_state.detection_history])
                st.metric("⚡ Avg Threat Score", f"{avg_score:.2f}")
            else:
                st.metric("⚡ Avg Threat Score", "0.00")

        # Live monitoring
        if st.session_state.monitoring:
            # Placeholder for real-time updates
            status_placeholder = st.empty()
            gauges_placeholder = st.empty()
            log_placeholder = st.empty()

            # Process multiple packets before rerun (smoother animation)
            packets_per_batch = 1

            for _ in range(packets_per_batch):
                # Simulate traffic
                time.sleep(1.0 / traffic_rate)

                # Decide if this packet is attack
                is_attack_packet = np.random.random() < (attack_prob / 100)

            # Get random normal samples
            normal_idx = {
                layer: np.where(samples[layer]['y'] == 0)[0][0] if (samples[layer]['y'] == 0).any() else 0
                for layer in ['network', 'iot', 'linux', 'windows']
            }

            detections = {}

            if is_attack_packet:
                # Random attack on 1-3 layers
                num_attack_layers = np.random.randint(1, 4)
                attack_layers = np.random.choice(['network', 'iot', 'linux', 'windows'],
                                                 size=num_attack_layers, replace=False)

                for layer in ['network', 'iot', 'linux', 'windows']:
                    normal_sample = samples[layer]['X'][normal_idx[layer]]

                    if layer in attack_layers:
                        noise = np.random.uniform(2.0, 4.0)
                        attack_sample = create_synthetic_attack(normal_sample, noise_level=noise)
                        detections[layer] = ids.detect_layer(layer, attack_sample)
                    else:
                        detections[layer] = ids.detect_layer(layer, normal_sample)
            else:
                # All normal
                for layer in ['network', 'iot', 'linux', 'windows']:
                    features = samples[layer]['X'][normal_idx[layer]]
                    detections[layer] = ids.detect_layer(layer, features)

            # Fusion decision
            fusion = ids.fusion_decision(detections)

            # Update stats
            st.session_state.packet_count += 1
            if fusion['is_attack']:
                st.session_state.attack_count += 1

            # Log entry
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            attacked_layers = [ids.layers[k]['name'] for k, v in detections.items() if v['is_attack']]

            if fusion['is_attack']:
                log_entry = f"[{timestamp}] 🚨 ATTACK DETECTED | Score: {fusion['score']:.2f} | Layers: {', '.join(attacked_layers)}"
            else:
                log_entry = f"[{timestamp}] ✅ Normal Traffic | Score: {fusion['score']:.2f}"

            st.session_state.traffic_log.insert(0, log_entry)
            st.session_state.traffic_log = st.session_state.traffic_log[:50]  # Keep last 50

            # Save to history
            st.session_state.detection_history.append({
                'timestamp': datetime.datetime.now(),
                'fusion_score': fusion['score'],
                'threat_level': fusion['threat_level'],
                'is_attack': fusion['is_attack']
            })

            # Display status
            if fusion['is_attack']:
                status_placeholder.markdown(
                    f'<div class="status-attack">🚨 ATTACK DETECTED! Threat Level: {fusion["threat_level"]} | Score: {fusion["score"]:.2f}</div>',
                    unsafe_allow_html=True
                )
            else:
                status_placeholder.markdown(
                    f'<div class="status-normal">✅ SYSTEM NORMAL | Score: {fusion["score"]:.2f}</div>',
                    unsafe_allow_html=True
                )

            # Display gauges
            with gauges_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    fig = plot_mini_gauge(
                        detections['network']['error'],
                        detections['network']['threshold'],
                        "Network",
                        detections['network']['is_attack']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = plot_mini_gauge(
                        detections['iot']['error'],
                        detections['iot']['threshold'],
                        "IoT/SCADA",
                        detections['iot']['is_attack']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    fig = plot_mini_gauge(
                        detections['linux']['error'],
                        detections['linux']['threshold'],
                        "Linux",
                        detections['linux']['is_attack']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col4:
                    fig = plot_mini_gauge(
                        detections['windows']['error'],
                        detections['windows']['threshold'],
                        "Windows",
                        detections['windows']['is_attack']
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Display traffic log
            with log_placeholder.container():
                st.subheader("📜 Traffic Log")
                log_html = '<div class="traffic-log">'
                for entry in st.session_state.traffic_log:
                    log_html += f"{entry}<br>"
                log_html += '</div>'
                st.markdown(log_html, unsafe_allow_html=True)

            # Auto-refresh with delay to avoid flickering
            time.sleep(0.5)  # Increased delay for smoother experience
            st.rerun()

        else:
            st.info("👈 Click 'Start' in the sidebar to begin live monitoring")

            # Show historical data if available
            if st.session_state.detection_history:
                st.subheader("📈 Detection History")

                df = pd.DataFrame(st.session_state.detection_history)

                fig = go.Figure()

                # Color by attack status
                colors = ['red' if x else 'green' for x in df['is_attack']]

                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['fusion_score'],
                    mode='lines+markers',
                    name='Fusion Score',
                    line=dict(color='blue', width=2),
                    marker=dict(
                        size=8,
                        color=colors,
                        line=dict(width=1, color='white')
                    )
                ))

                fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                             annotation_text="Attack Threshold")

                fig.update_layout(
                    title="Real-time Threat Score",
                    xaxis_title="Time",
                    yaxis_title="Fusion Score",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

    elif mode == "📁 Upload CSV":
        if 'csv_results' in st.session_state:
            results = st.session_state.csv_results
            layer = results['layer']
            features = results['features']
            labels = results['labels']

            st.subheader(f"📊 Analysis Results - {ids.layers[layer]['name']}")

            # Analyze all samples
            detections = []
            for i, feature_row in enumerate(features):
                detection = ids.detect_layer(layer, feature_row)
                detection['sample_id'] = i
                if labels is not None:
                    detection['true_label'] = labels[i]
                detections.append(detection)

            # Summary stats
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📦 Total Samples", len(detections))

            with col2:
                attack_detected = sum(1 for d in detections if d['is_attack'])
                st.metric("🚨 Attacks Detected", attack_detected)

            with col3:
                detection_rate = (attack_detected / len(detections)) * 100
                st.metric("📊 Detection Rate", f"{detection_rate:.1f}%")

            # Results table
            st.subheader("📋 Detailed Results")

            df = pd.DataFrame(detections)
            df['Status'] = df['is_attack'].apply(lambda x: '🔴 ATTACK' if x else '🟢 NORMAL')

            display_cols = ['sample_id', 'Status', 'error', 'threshold', 'severity']
            if labels is not None:
                display_cols.insert(2, 'true_label')

            st.dataframe(df[display_cols], use_container_width=True)

            # Distribution plot
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=[d['error'] for d in detections if not d['is_attack']],
                name='Normal',
                marker_color='green',
                opacity=0.7
            ))

            fig.add_trace(go.Histogram(
                x=[d['error'] for d in detections if d['is_attack']],
                name='Attack',
                marker_color='red',
                opacity=0.7
            ))

            fig.add_vline(
                x=detections[0]['threshold'],
                line_dash="dash",
                line_color="black",
                annotation_text="Threshold"
            )

            fig.update_layout(
                title="Reconstruction Error Distribution",
                xaxis_title="Error",
                yaxis_title="Count",
                barmode='overlay',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Upload a CSV file in the sidebar to analyze")

            with st.expander("ℹ️ CSV Format Guide"):
                st.markdown("""
                **Expected CSV format:**
                - One row per sample
                - Features as columns (numeric values)
                - Optional label column (will be auto-detected)

                **Example:**
                ```
                feature1,feature2,feature3,...,label
                0.123,0.456,0.789,...,0
                0.234,0.567,0.890,...,1
                ```

                **Supported label column names:** label, Label, attack, Attack, class, Class
                """)

    else:  # Manual Test
        if 'manual_test' in st.session_state:
            scenario = st.session_state.manual_test

            # Get normal samples
            normal_idx = {
                layer: np.where(samples[layer]['y'] == 0)[0][0] if (samples[layer]['y'] == 0).any() else 0
                for layer in ['network', 'iot', 'linux', 'windows']
            }

            detections = {}

            if scenario == "Normal Traffic":
                for layer in ['network', 'iot', 'linux', 'windows']:
                    features = samples[layer]['X'][normal_idx[layer]]
                    detections[layer] = ids.detect_layer(layer, features)

            elif scenario == "Single Layer Attack":
                network_normal = samples['network']['X'][normal_idx['network']]
                network_attack = create_synthetic_attack(network_normal, noise_level=2.5)
                detections['network'] = ids.detect_layer('network', network_attack)

                for layer in ['iot', 'linux', 'windows']:
                    features = samples[layer]['X'][normal_idx[layer]]
                    detections[layer] = ids.detect_layer(layer, features)

            else:  # Multi-Layer Attack
                for layer in ['network', 'iot', 'linux', 'windows']:
                    normal_sample = samples[layer]['X'][normal_idx[layer]]
                    noise = 3.0 if layer in ['network', 'iot'] else 2.5
                    attack_sample = create_synthetic_attack(normal_sample, noise_level=noise)
                    detections[layer] = ids.detect_layer(layer, attack_sample)

            # Fusion decision
            fusion = ids.fusion_decision(detections)

            # Display results
            if fusion['is_attack']:
                st.markdown(
                    f'<div class="status-attack">🚨 ATTACK DETECTED! Threat Level: {fusion["threat_level"]} | Score: {fusion["score"]:.2f}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="status-normal">✅ SYSTEM NORMAL | Score: {fusion["score"]:.2f}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # Layer details
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Network & IoT")

                for layer in ['network', 'iot']:
                    status = "🔴 ATTACK" if detections[layer]['is_attack'] else "🟢 NORMAL"
                    st.metric(
                        ids.layers[layer]['name'],
                        status,
                        f"Error: {detections[layer]['error']:.6f}"
                    )

            with col2:
                st.subheader("📊 System Layers")

                for layer in ['linux', 'windows']:
                    status = "🔴 ATTACK" if detections[layer]['is_attack'] else "🟢 NORMAL"
                    st.metric(
                        ids.layers[layer]['name'],
                        status,
                        f"Error: {detections[layer]['error']:.6f}"
                    )

        else:
            st.info("👈 Select a scenario and click 'Run Test' in the sidebar")

if __name__ == "__main__":
    main()
