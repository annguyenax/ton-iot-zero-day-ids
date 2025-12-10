"""
IMPROVED Dashboard for Zero-Day IoT Attack Detection System
- Real samples from test data (not random!)
- CSV auto-detection by feature count
- Detailed logging with feature values, errors, thresholds
- Clear differentiation between normal and attack traffic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import joblib
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
import tensorflow as tf

# Clear Keras session
keras.backend.clear_session()

# Import modules
from data_loader import load_ton_iot_data
from preprocessor import preprocess_data

# Page config
st.set_page_config(
    page_title="Zero-Day IoT IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-critical {
        background-color: #ff4b4b;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffa500;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .alert-normal {
        background-color: #00c853;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .log-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# Load models AND test samples
@st.cache_resource
def load_models_and_samples():
    """Load all trained models, scalers, AND real test samples"""
    keras.backend.clear_session()

    models = {}
    scalers = {}
    thresholds = {}
    test_samples = {}  # NEW: Load real test samples

    layers = ['network', 'iot', 'linux', 'windows']

    for layer in layers:
        try:
            model_path = f'../models/unsupervised/{layer}_autoencoder.h5'
            scaler_path = f'../models/unsupervised/{layer}_scaler.pkl'
            threshold_path = f'../models/unsupervised/{layer}_threshold.pkl'
            samples_X_path = f'../models/unsupervised/{layer}_samples_X.npy'
            samples_y_path = f'../models/unsupervised/{layer}_samples_y.npy'

            if not os.path.exists(model_path):
                st.warning(f"Model file not found: {model_path}")
                continue

            # Load model, scaler, threshold
            models[layer] = keras.models.load_model(model_path, compile=False)
            models[layer].compile(optimizer='adam', loss='mse')
            scalers[layer] = joblib.load(scaler_path)
            thresholds[layer] = joblib.load(threshold_path)

            # Load REAL test samples
            X_test = np.load(samples_X_path)
            y_test = np.load(samples_y_path)
            test_samples[layer] = {'X': X_test, 'y': y_test}

            st.success(f"✓ Loaded {layer} model + {len(X_test)} test samples", icon="✅")
        except Exception as e:
            st.error(f"Error loading {layer}: {str(e)}")

    return models, scalers, thresholds, test_samples


def detect_anomaly_detailed(sample, model, scaler, threshold, layer_name):
    """Detect anomaly with DETAILED logging"""
    try:
        # Scale
        sample_scaled = scaler.transform(sample.reshape(1, -1))

        # Predict (reconstruction)
        reconstructed = model.predict(sample_scaled, verbose=0)

        # Calculate reconstruction error (MSE)
        error = np.mean(np.power(sample_scaled - reconstructed, 2))

        # Detect
        is_attack = error > threshold
        confidence = min(100, (error / threshold) * 100) if is_attack else max(0, 100 - (error / threshold) * 100)

        # Detailed info for logging
        return {
            'is_attack': bool(is_attack),
            'error': float(error),
            'threshold': float(threshold),
            'confidence': float(confidence),
            'layer': layer_name,
            'n_features': len(sample),
            'sample_scaled': sample_scaled[0],  # For detailed logging
            'reconstructed': reconstructed[0],   # For detailed logging
        }
    except Exception as e:
        st.error(f"Detection error in {layer_name}: {e}")
        return None


def fusion_engine(results):
    """Multi-layer fusion for final decision"""
    # Count layers that detected attack
    attack_count = sum([1 for r in results.values() if r and r['is_attack']])

    # Average confidence
    confidences = [r['confidence'] for r in results.values() if r]
    avg_confidence = np.mean(confidences) if confidences else 0

    # Fusion logic
    if attack_count >= 3:
        threat_level = "CRITICAL"
        color = "red"
    elif attack_count >= 2:
        threat_level = "HIGH"
        color = "orange"
    elif attack_count >= 1:
        threat_level = "MEDIUM"
        color = "yellow"
    else:
        threat_level = "NORMAL"
        color = "green"

    return {
        'threat_level': threat_level,
        'attack_count': attack_count,
        'confidence': avg_confidence,
        'color': color
    }


def create_gauge_chart(value, title, threshold=50):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        delta={'reference': threshold},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def format_detection_log(sample_info, results, is_actual_attack=None):
    """Format detailed detection log"""
    log = f"""
{'='*80}
🕐 TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📦 PACKET INFO:
   Sample Type: {'🔴 ATTACK' if is_actual_attack else '🟢 NORMAL' if is_actual_attack is not None else '❓ UNKNOWN'}
   Total Features: {results[list(results.keys())[0]]['n_features'] if results else 'N/A'}

{'='*80}
🔍 LAYER-BY-LAYER DETECTION:
{'='*80}
"""

    for layer, result in results.items():
        if result:
            status = "🔴 ATTACK DETECTED" if result['is_attack'] else "🟢 NORMAL"
            log += f"""
📊 {layer.upper()} LAYER:
   Status: {status}
   Reconstruction Error: {result['error']:.6f}
   Detection Threshold: {result['threshold']:.6f}
   Confidence: {result['confidence']:.1f}%
   Features: {result['n_features']}
   Error vs Threshold: {result['error']/result['threshold']:.2f}x
"""

    # Fusion result
    fusion = fusion_engine(results)
    log += f"""
{'='*80}
🎯 FUSION ENGINE RESULT:
{'='*80}
   Final Threat Level: {fusion['threat_level']}
   Layers Detected Attack: {fusion['attack_count']}/4
   Average Confidence: {fusion['confidence']:.1f}%

"""
    return log


def auto_detect_layer(n_features):
    """Auto-detect layer based on feature count"""
    layer_features = {
        'network': 40,
        'iot': 5,
        'linux': 12,
        'windows': 52
    }

    for layer, features in layer_features.items():
        if n_features == features:
            return layer

    return None


# ==================== MAIN ====================
def main():
    st.markdown('<div class="main-header">🛡️ Zero-Day IoT Attack Detection System</div>', unsafe_allow_html=True)

    # Load models and test samples
    models, scalers, thresholds, test_samples = load_models_and_samples()

    if not models:
        st.error("Failed to load models. Please train models first!")
        return

    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["📊 Real-time Monitoring", "📁 CSV Upload & Analysis", "🧪 Manual Testing"]
    )

    # ==================== REAL-TIME MONITORING (IMPROVED) ====================
    if mode == "📊 Real-time Monitoring":
        st.header("📊 Real-time Network Monitoring with REAL DATA")

        st.info("✨ **Using REAL test samples** (not random!): Mix of normal & attack traffic")

        # Select primary layer for simulation
        simulation_layer = st.selectbox("Select Traffic Layer", ['network', 'iot', 'linux', 'windows'], index=0)

        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            simulation_speed = st.slider("Speed (packets/sec)", 1, 10, 2)
        with col2:
            attack_ratio = st.slider("Attack Ratio (%)", 0, 100, 30)
        with col3:
            enable_simulation = st.checkbox("▶️ Start Simulation", value=False)

        # Placeholders
        metrics_placeholder = st.empty()
        gauges_placeholder = st.empty()
        alerts_placeholder = st.empty()
        log_placeholder = st.empty()
        history_placeholder = st.empty()

        # Initialize session state
        if 'packet_history' not in st.session_state:
            st.session_state.packet_history = []

        if enable_simulation:
            # Get test samples from selected layer
            X_test = test_samples[simulation_layer]['X']
            y_test = test_samples[simulation_layer]['y']

            # Separate normal and attack samples
            normal_indices = np.where(y_test == 0)[0]
            attack_indices = np.where(y_test == 1)[0]

            if len(normal_indices) == 0 or len(attack_indices) == 0:
                st.error("Not enough samples in test data!")
                return

            # Simulation loop
            for packet_num in range(20):  # Simulate 20 packets
                # Decide if this packet is attack based on attack_ratio
                is_attack_actual = np.random.rand() < (attack_ratio / 100)

                # Select a REAL sample
                if is_attack_actual and len(attack_indices) > 0:
                    idx = np.random.choice(attack_indices)
                    sample = X_test[idx]
                    actual_label = "ATTACK"
                else:
                    idx = np.random.choice(normal_indices)
                    sample = X_test[idx]
                    actual_label = "NORMAL"

                # Detect on PRIMARY layer only (for this sample's features)
                result_primary = detect_anomaly_detailed(
                    sample,
                    models[simulation_layer],
                    scalers[simulation_layer],
                    thresholds[simulation_layer],
                    simulation_layer
                )

                # For other layers: use subset/padding to make it work (simplified)
                results = {simulation_layer: result_primary}
                for other_layer in models.keys():
                    if other_layer != simulation_layer:
                        # Use zeros for other layers (simplified simulation)
                        n_features_other = models[other_layer].input_shape[1]
                        sample_other = np.zeros(n_features_other)
                        results[other_layer] = detect_anomaly_detailed(
                            sample_other,
                            models[other_layer],
                            scalers[other_layer],
                            thresholds[other_layer],
                            other_layer
                        )

                # Fusion
                fusion = fusion_engine(results)

                # Store in history
                st.session_state.packet_history.append({
                    'timestamp': datetime.now(),
                    'threat_level': fusion['threat_level'],
                    'confidence': fusion['confidence'],
                    'actual_label': actual_label,
                    'predicted_label': 'ATTACK' if result_primary['is_attack'] else 'NORMAL',
                    'results': results
                })

                # Keep last 50
                if len(st.session_state.packet_history) > 50:
                    st.session_state.packet_history.pop(0)

                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("📦 Total Packets", len(st.session_state.packet_history))
                    with col2:
                        attacks_detected = sum([1 for p in st.session_state.packet_history
                                              if p['predicted_label'] == 'ATTACK'])
                        st.metric("🚨 Attacks Detected", attacks_detected)
                    with col3:
                        actual_attacks = sum([1 for p in st.session_state.packet_history
                                             if p['actual_label'] == 'ATTACK'])
                        st.metric("🔴 Actual Attacks", actual_attacks)
                    with col4:
                        st.metric("🎯 Current Threat", fusion['threat_level'])
                    with col5:
                        st.metric("📊 Confidence", f"{fusion['confidence']:.1f}%")

                # Gauges
                with gauges_placeholder.container():
                    cols = st.columns(4)
                    for idx, (layer, result) in enumerate(results.items()):
                        if result:
                            with cols[idx]:
                                st.plotly_chart(
                                    create_gauge_chart(result['confidence'], layer.upper(), 70),
                                    width='stretch',
                                    key=f"gauge_{layer}_{len(st.session_state.packet_history)}"
                                )

                # Alert
                with alerts_placeholder.container():
                    if fusion['threat_level'] == 'CRITICAL':
                        st.markdown(f'<div class="alert-critical">🚨 CRITICAL THREAT! Confidence: {fusion["confidence"]:.1f}%</div>',
                                  unsafe_allow_html=True)
                    elif fusion['threat_level'] == 'HIGH':
                        st.markdown(f'<div class="alert-warning">⚠️ HIGH THREAT! Confidence: {fusion["confidence"]:.1f}%</div>',
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-normal">✓ Status: NORMAL</div>',
                                  unsafe_allow_html=True)

                # Detailed log
                with log_placeholder.container():
                    st.subheader("📋 Detection Log (Latest Packet)")
                    log_text = format_detection_log(
                        {'actual': actual_label},
                        results,
                        is_actual_attack=is_attack_actual
                    )
                    st.markdown(f'<div class="log-box"><pre>{log_text}</pre></div>', unsafe_allow_html=True)

                # History chart
                if len(st.session_state.packet_history) > 1:
                    with history_placeholder.container():
                        df_history = pd.DataFrame([
                            {
                                'Time': i,
                                'Confidence': p['confidence'],
                                'Actual': p['actual_label'],
                                'Predicted': p['predicted_label']
                            } for i, p in enumerate(st.session_state.packet_history)
                        ])

                        fig = px.line(df_history, x='Time', y='Confidence',
                                    title='Threat Confidence Over Time',
                                    markers=True)

                        # Add markers for actual vs predicted
                        fig.add_scatter(x=df_history['Time'],
                                      y=[50 if p == 'ATTACK' else 10 for p in df_history['Actual']],
                                      mode='markers',
                                      name='Actual (Attack=50, Normal=10)',
                                      marker=dict(size=8, color='red'))

                        st.plotly_chart(fig, width='stretch')

                time.sleep(1.0 / simulation_speed)

    # ==================== CSV UPLOAD (IMPROVED) ====================
    elif mode == "📁 CSV Upload & Analysis":
        st.header("📁 CSV Upload & Intelligent Analysis")

        st.info("✨ **Auto-detects layer** based on feature count after preprocessing!")

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write(f"📊 Loaded {len(df)} samples with {df.shape[1]} columns")
            st.dataframe(df.head())

            if st.button("🔍 Run Detection"):
                with st.spinner("Analyzing..."):
                    try:
                        # Preprocess
                        X, y_attack, _, _ = preprocess_data(df, label_col='type' if 'type' in df.columns else 'label' if 'label' in df.columns else None)

                        st.success(f"✓ Preprocessed: {X.shape[0]} samples, {X.shape[1]} features")

                        # AUTO-DETECT LAYER
                        detected_layer = auto_detect_layer(X.shape[1])

                        if detected_layer is None:
                            st.error(f"❌ Cannot detect layer! Feature count {X.shape[1]} doesn't match any layer:")
                            st.write("- Network: 40 features")
                            st.write("- IoT: 5 features")
                            st.write("- Linux: 12 features")
                            st.write("- Windows: 52 features")
                            return

                        st.success(f"🎯 **Detected Layer: {detected_layer.upper()}** (matches {X.shape[1]} features)")

                        # Get model for detected layer
                        model = models[detected_layer]
                        scaler = scalers[detected_layer]
                        threshold = thresholds[detected_layer]

                        # Analyze (limit to first 500 for speed)
                        max_samples = min(500, len(X))
                        results_list = []

                        progress_bar = st.progress(0)
                        for i in range(max_samples):
                            sample = X.iloc[i].values
                            result = detect_anomaly_detailed(sample, model, scaler, threshold, detected_layer)
                            if result:
                                results_list.append(result)
                            progress_bar.progress((i + 1) / max_samples)

                        # Analysis results
                        st.success(f"✅ Analyzed {len(results_list)} samples")

                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            attacks_detected = sum([1 for r in results_list if r['is_attack']])
                            st.metric("🚨 Attacks Detected", attacks_detected)
                        with col2:
                            normal_detected = len(results_list) - attacks_detected
                            st.metric("🟢 Normal", normal_detected)
                        with col3:
                            avg_error = np.mean([r['error'] for r in results_list])
                            st.metric("📊 Avg Error", f"{avg_error:.4f}")
                        with col4:
                            st.metric("🎯 Threshold", f"{threshold:.4f}")

                        # Distribution chart
                        errors = [r['error'] for r in results_list]
                        labels = ['Attack' if r['is_attack'] else 'Normal' for r in results_list]

                        df_results = pd.DataFrame({
                            'Reconstruction Error': errors,
                            'Label': labels
                        })

                        fig = px.histogram(df_results, x='Reconstruction Error', color='Label',
                                         title='Reconstruction Error Distribution',
                                         nbins=50, barmode='overlay')
                        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                                    annotation_text=f"Threshold={threshold:.4f}")
                        st.plotly_chart(fig, width='stretch')

                        # Detailed results table
                        st.subheader("📋 Detailed Results (First 20)")
                        df_table = pd.DataFrame([
                            {
                                'Sample': i,
                                'Error': f"{r['error']:.6f}",
                                'Threshold': f"{r['threshold']:.6f}",
                                'Predicted': '🔴 Attack' if r['is_attack'] else '🟢 Normal',
                                'Confidence': f"{r['confidence']:.1f}%"
                            } for i, r in enumerate(results_list[:20])
                        ])
                        st.dataframe(df_table)

                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        import traceback
                        st.code(traceback.format_exc())

    # ==================== MANUAL TESTING (IMPROVED) ====================
    elif mode == "🧪 Manual Testing":
        st.header("🧪 Manual Attack Testing with Detailed Analysis")

        st.info("✨ **Test individual samples** with full feature analysis")

        # Select layer
        layer = st.selectbox("Select Layer", ['network', 'iot', 'linux', 'windows'])

        try:
            X_test = test_samples[layer]['X']
            y_test = test_samples[layer]['y']

            # Sample selection
            sample_idx = st.slider("Select Sample", 0, len(X_test) - 1, 0)
            sample = X_test[sample_idx]
            is_attack_actual = y_test[sample_idx] == 1

            st.write(f"**Sample #{sample_idx}**: {'🔴 ATTACK' if is_attack_actual else '🟢 NORMAL'} (actual label)")
            st.write(f"**Features**: {len(sample)}")

            # Detect
            result = detect_anomaly_detailed(
                sample,
                models[layer],
                scalers[layer],
                thresholds[layer],
                layer
            )

            if result:
                # Display result
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Predicted: {'🔴 ATTACK' if result['is_attack'] else '🟢 NORMAL'}")
                with col2:
                    correct = (result['is_attack'] == is_attack_actual)
                    st.markdown(f"### {'✅ CORRECT' if correct else '❌ WRONG'}")

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reconstruction Error", f"{result['error']:.6f}")
                with col2:
                    st.metric("Threshold", f"{result['threshold']:.6f}")
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")

                # Gauge chart
                st.plotly_chart(create_gauge_chart(result['confidence'], layer.upper(), 70), width='stretch')

                # Detailed feature analysis
                st.subheader("🔬 Feature Analysis")

                # Show first 10 features (original vs reconstructed)
                st.write("**Sample Features (first 10):**")
                feature_df = pd.DataFrame({
                    'Feature #': range(min(10, len(sample))),
                    'Original Value': sample[:10],
                })
                st.dataframe(feature_df)

                # Error breakdown
                st.subheader("📊 Error Breakdown")
                feature_errors = np.power(result['sample_scaled'] - result['reconstructed'], 2)
                top_error_indices = np.argsort(feature_errors)[-10:][::-1]  # Top 10 error features

                error_df = pd.DataFrame({
                    'Feature #': top_error_indices,
                    'Squared Error': feature_errors[top_error_indices],
                    'Contribution %': (feature_errors[top_error_indices] / result['error'] * 100)
                })
                st.dataframe(error_df)

                # Full log
                st.subheader("📋 Full Detection Log")
                log_text = format_detection_log(
                    {'sample_idx': sample_idx},
                    {layer: result},
                    is_actual_attack=is_attack_actual
                )
                st.markdown(f'<div class="log-box"><pre>{log_text}</pre></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
