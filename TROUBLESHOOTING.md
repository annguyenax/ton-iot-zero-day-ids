# =' TROUBLESHOOTING GUIDE

Common issues và cách fix cho Zero-Day IoT Attack Detection System.

---

## =Ë TABLE OF CONTENTS

1. [Installation Issues](#installation-issues)
2. [Training Issues](#training-issues)
3. [Dashboard Issues](#dashboard-issues)
4. [Docker Issues](#docker-issues)
5. [Performance Issues](#performance-issues)
6. [Data Issues](#data-issues)

---

## =4 INSTALLATION ISSUES

### Issue 1: `pip install` fails vÛi error vÁ TensorFlow

**Error**:
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0
```

**Cause**: Python version không t°¡ng thích

**Solution**:
```bash
# Check Python version
python --version

# TensorFlow 2.15 c§n Python 3.9-3.11
# N¿u Python < 3.9 ho·c > 3.11:

# Option 1: Cài Python 3.10
# Download të python.org

# Option 2: Dùng TensorFlow version khác
pip install tensorflow==2.13.0  # Cho Python 3.8
pip install tensorflow==2.16.1  # Cho Python 3.12
```

---

### Issue 2: ModuleNotFoundError sau khi install

**Error**:
```
ModuleNotFoundError: No module named 'streamlit'
```

**Cause**: Virtual environment không active ho·c install sai environment

**Solution**:
```bash
# Verify venv is activated
# Prompt should show (.venv) at beginning

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate

# Reinstall
pip install -r requirements_minimal.txt

# Verify
pip list | grep streamlit
```

---

### Issue 3: Permission denied khi install trên Linux

**Error**:
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solution**:
```bash
# DON'T use sudo pip!

# Use venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_minimal.txt

# OR install user-wide (not recommended)
pip install --user -r requirements_minimal.txt
```

---

## >à TRAINING ISSUES

### Issue 4: Training r¥t ch­m (>2 hours)

**Symptoms**: Training 1 epoch m¥t 10+ phút

**Causes & Solutions**:

**Cause 1: No GPU**
```bash
# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU, reduce dataset size
# Edit train_unsupervised.py line 305:
layers = [
    ('network', ..., 50000),  # Was: None (211K)
    ('iot', ..., 20000),
    ('linux', ..., 20000),
    ('windows', ..., 20000),
]
```

**Cause 2: Large batch size**
```python
# Edit train_unsupervised.py line 134:
batch_size=128  # Was: 256
```

**Cause 3: Too many epochs**
```python
# Edit train_unsupervised.py line 133:
epochs=50  # Was: 100
```

---

### Issue 5: Training crashes vÛi OutOfMemoryError

**Error**:
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor
```

**Solution**:
```python
# Edit train_unsupervised.py, add at top:
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# OR reduce batch size
batch_size=64  # Was: 256

# OR limit TensorFlow memory
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
    )
```

---

### Issue 6: Models không save

**Error**:
```
PermissionError: [Errno 13] Permission denied: 'models/unsupervised/network_autoencoder.h5'
```

**Solution**:
```bash
# Check directory exists
mkdir -p models/unsupervised

# Check write permission
ls -la models/unsupervised/

# Fix permission (Linux)
chmod -R 755 models/

# Windows: Right-click folder ’ Properties ’ Security ’ Edit ’ Allow Full Control
```

---

## =¥ DASHBOARD ISSUES

### Issue 7: Dashboard không start - Port already in use

**Error**:
```
Error: Port 8501 is already in use
```

**Solution**:
```bash
# Find process using port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8501
kill -9 <PID>

# Or use different port
streamlit run dashboard_zeroday.py --server.port 8502
```

---

### Issue 8: Dashboard loads nh°ng hiÃn thË l×i "Models not found"

**Error** (trên dashboard):
```
L Error loading models: No module named 'models'
```

**Cause**: Ch¡y dashboard të wrong directory

**Solution**:
```bash
# PH¢I ch¡y të src/ directory
cd src
streamlit run dashboard_zeroday.py

# NOT from root:
# streamlit run src/dashboard_zeroday.py  # L WRONG
```

---

### Issue 9: Duplicate key error trong Real-time mode

**Error**:
```
StreamlitDuplicateElementKey: There are multiple elements with the same key='gauge_iot_50'
```

**Cause**: Bug ã fix trong latest version

**Solution**:
```bash
# Pull latest code
git pull origin main

# Or manually fix dashboard_zeroday.py line 430:
# OLD:
key=f"gauge_{layer}_{len(st.session_state.packet_history)}"

# NEW:
key=f"gauge_{layer}_{idx}_{iteration}"
```

---

### Issue 10: CSV upload báo l×i "Feature mismatch"

**Error**:
```
Error: Expected 52 features, got 40
```

**Cause**: Uploaded wrong CSV cho wrong layer

**Solution**:
Dashboard auto-detects layer based on features:
- 40 features ’ Network
- 5 features ’ IoT
- 12 features ’ Linux
- 52 features ’ Windows

**Verify CSV**:
```bash
# Check number of columns
head -1 your_file.csv | awk -F',' '{print NF}'

# Should match one of: 40, 5, 12, 52
```

**If custom CSV**:
Make sure columns match TON_IoT dataset features exactly!

---

### Issue 11: Dashboard ch­m, lag khi Real-time mode

**Symptoms**: Dashboard freezes, metrics không update

**Solutions**:

1. **Reduce simulation speed**:
   - ChÍn 1x thay vì 5x

2. **Close other browser tabs**:
   - Streamlit consumes RAM

3. **Reduce history length** (edit `dashboard_zeroday.py`):
```python
# Line ~365:
if len(st.session_state.packet_history) > 50:  # Was: 100
    st.session_state.packet_history = st.session_state.packet_history[-50:]
```

4. **Use production mode**:
```bash
streamlit run dashboard_zeroday.py --server.runOnSave false
```

---

## =3 DOCKER ISSUES

### Issue 12: Docker build fails - "No such file or directory"

**Error**:
```
COPY src/ ./src/
ERROR: failed to compute cache key: "/src" not found
```

**Cause**: Building from wrong directory

**Solution**:
```bash
# MUST build from project root
cd d:\Zero-day-IoT-Attack-Detection

# Then:
docker build -t zero-day-iot-ids .

# NOT from src/:
# cd src && docker build ...  # L WRONG
```

---

### Issue 13: Container starts but dashboard not accessible

**Symptoms**:
- `docker ps` shows container running
- But http://localhost:8501 không accessible

**Solutions**:

1. **Check port mapping**:
```bash
docker ps
# Should see: 0.0.0.0:8501->8501/tcp

# If not, restart with correct port:
docker run -p 8501:8501 zero-day-iot-ids
```

2. **Check firewall** (Windows):
```powershell
# Allow port 8501
New-NetFirewallRule -DisplayName "Streamlit" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```

3. **Check container logs**:
```bash
docker logs -f <container_id>

# Look for errors like:
# - ModuleNotFoundError
# - PermissionError
# - Port already in use
```

---

### Issue 14: Models not loading trong Docker

**Error** (trong container logs):
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/unsupervised/network_autoencoder.h5'
```

**Cause**: Volume không mount úng

**Solution**:
```bash
# docker-compose.yml:
volumes:
  - ./models:/app/models:ro  # :ro = read-only

# Verify volume mounted:
docker exec <container_id> ls -la /app/models/unsupervised/

# Should see .h5, .pkl, .npy files
```

**If files missing**:
```bash
# Copy models vào container:
docker cp models/ <container_id>:/app/
```

---

## ¡ PERFORMANCE ISSUES

### Issue 15: Detection r¥t ch­m (>5s per sample)

**Symptoms**: Manual testing m¥t 5-10s m×i prediction

**Solutions**:

1. **Batch predictions**:
```python
# Instead of:
for sample in samples:
    model.predict(sample)  # Slow

# Use:
model.predict(np.array(samples))  # Fast
```

2. **Compile model**:
```python
# Edit train_unsupervised.py line ~110:
model.compile(optimizer='adam', loss='mse', jit_compile=True)  # Add jit_compile
```

3. **Use GPU**:
```bash
# Install GPU version
pip uninstall tensorflow
pip install tensorflow-gpu==2.15.0
```

---

### Issue 16: High False Positive rate (>25%)

**Symptoms**: Too many false alarms

**Solutions**:

1. **Increase thresholds** (edit `train_unsupervised.py`):
```python
# Network (line 184):
threshold = np.percentile(clean_errors, 85)  # Was: 82, higher = less FP

# Linux (line 190):
threshold = mean_error + 1.5 * std_error  # Was: 1.2, higher = less FP
```

2. **Retrain on more normal data**:
```python
# Increase normal traffic ratio in dataset
# Or collect more normal samples
```

3. **Use fusion with higher threshold**:
```python
# dashboard_zeroday.py line ~400:
# Only alert if 2+ layers detect (instead of 1+)
```

---

### Issue 17: Low Detection rate (<80%)

**Symptoms**: Missing too many attacks

**Solutions**:

1. **Decrease thresholds** (opposite of Issue 16):
```python
# Network (line 184):
threshold = np.percentile(clean_errors, 80)  # Was: 82, lower = more sensitive

# Windows (line 198):
threshold = np.percentile(clean_errors, 97)  # Was: 99
```

2. **Train longer**:
```python
# train_unsupervised.py line 133:
epochs=150  # Was: 100
```

3. **Increase model capacity**:
```python
# train_unsupervised.py line ~80:
# Network example:
encoder = Dense(30, activation='relu')(input_layer)  # Was: 20
encoded = Dense(15, activation='relu')(encoder)      # Was: 10
```

---

## =Ê DATA ISSUES

### Issue 18: Dataset download fails

**Error**:
```
urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>
```

**Solutions**:

1. **Manual download**:
   - Go to: https://research.unsw.edu.au/projects/toniot-datasets
   - Download manually
   - Extract to `data/Train_Test_datasets/`

2. **Use alternate mirror** (if available)
3. **Check internet connection**

---

### Issue 19: CSV corrupted ho·c wrong format

**Error**:
```
pandas.errors.ParserError: Error tokenizing data
```

**Solutions**:

1. **Check CSV encoding**:
```bash
file -i your_file.csv
# Should be: text/plain; charset=utf-8

# Convert if needed:
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

2. **Check delimiter**:
```python
# If not comma-separated:
df = pd.read_csv('file.csv', delimiter=';')  # Or '\t' for tab
```

3. **Handle missing values**:
```python
df = pd.read_csv('file.csv', na_values=['?', 'N/A', 'null'])
df = df.fillna(0)  # Or df.dropna()
```

---

### Issue 20: Features không match TON_IoT format

**Error**:
```
KeyError: 'Column X not found in dataframe'
```

**Cause**: CSV columns khác vÛi expected features

**Solution**:

Check expected features cho m×i layer (trong `preprocessor.py`):

**Network (40 features)**:
```python
NETWORK_FEATURES = [
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
    'packet_len', 'flags', 'ttl', 'seq_num', 'ack_num',
    # ... (35 more)
]
```

**IoT (5 features)**:
```python
IOT_FEATURES = ['temperature', 'humidity', 'motion', 'light', 'pressure']
```

**Linux (12 features)**:
```python
LINUX_FEATURES = ['cpu_usage', 'mem_usage', 'disk_io', 'net_io', ...]
```

**Windows (52 features)**:
```python
WINDOWS_FEATURES = ['process_count', 'thread_count', 'registry_changes', ...]
```

**Map your CSV columns** to match these!

---

## <˜ STILL STUCK?

### Debugging Checklist:

- [ ] Check Python version (3.9-3.11)
- [ ] Verify venv is activated
- [ ] Verify all packages installed: `pip list`
- [ ] Check working directory: `pwd` or `cd`
- [ ] Check models exist: `ls models/unsupervised/`
- [ ] Check logs: `logs/` directory
- [ ] Try minimal example (single prediction)
- [ ] Restart Python kernel / terminal
- [ ] Restart Docker containers
- [ ] Clear cache: `rm -rf __pycache__/`

### Get More Help:

1. **Check GitHub Issues**: Similar problems may be already solved
2. **Enable debug logging**:
```python
# Add to top of dashboard_zeroday.py:
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. **Run in verbose mode**:
```bash
streamlit run dashboard_zeroday.py --logger.level=debug
```

4. **Check system resources**:
```bash
# CPU, RAM usage
top  # Linux/Mac
taskmgr  # Windows

# Disk space
df -h  # Linux/Mac
wmic logicaldisk get size,freespace  # Windows
```

5. **Create minimal reproducible example**:
```python
# test_minimal.py
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('models/unsupervised/network_autoencoder.h5')

# Test prediction
sample = np.random.randn(1, 40)
prediction = model.predict(sample)
print(f"Shape: {prediction.shape}")
print(" Model works!")
```

---

## =Ý REPORT A BUG

Khi report bug, please provide:

1. **Error message** (full traceback)
2. **Environment**:
   - OS: Windows/Linux/Mac
   - Python version: `python --version`
   - Package versions: `pip freeze`
3. **Steps to reproduce**
4. **Expected behavior**
5. **Actual behavior**
6. **Screenshots** (if applicable)

**Template**:
```markdown
## Bug Report

**Environment:**
- OS: Windows 11 Pro
- Python: 3.10.5
- TensorFlow: 2.15.0

**Steps to Reproduce:**
1. Run `python train_unsupervised.py`
2. Wait for epoch 10
3. Error occurs

**Error:**
```
<paste error here>
```

**Expected:** Training completes successfully
**Actual:** Crashes with OOM error

**Additional Context:**
- RAM: 8GB
- Dataset size: 211K samples
```

---

**Last Updated**: 2025-12-10
**Version**: 1.0
**Maintainer**: Claude Sonnet 4.5
