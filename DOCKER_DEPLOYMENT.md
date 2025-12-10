# 🐳 Docker Deployment Guide

Quick guide to deploy Zero-Day IoT Attack Detection System using Docker.

---

## 📋 Prerequisites

- Docker Desktop installed (Windows/Mac) or Docker Engine (Linux)
- Docker Compose (included in Docker Desktop)
- At least 4GB RAM available
- Port 8501 available

---

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

Access dashboard at: **http://localhost:8501**

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t zero-day-iot-ids .

# Run the container
docker run -d \
  --name zero-day-iot-ids \
  -p 8501:8501 \
  -v ./models:/app/models:ro \
  -v ./logs:/app/logs \
  zero-day-iot-ids

# View logs
docker logs -f zero-day-iot-ids

# Stop the container
docker stop zero-day-iot-ids
docker rm zero-day-iot-ids
```

---

## 🔧 Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - TF_CPP_MIN_LOG_LEVEL=3      # TensorFlow log level (0-3)
  - TF_ENABLE_ONEDNN_OPTS=0     # Disable oneDNN custom operations
  - PYTHONUNBUFFERED=1          # Enable real-time logs
```

### Volumes

- `./models:/app/models:ro` - Trained models (read-only)
- `./logs:/app/logs` - Application logs (read-write)

---

## 📊 Dashboard Features

Once running, access **http://localhost:8501** to use:

1. **📊 Real-time Monitoring**
   - Live network simulation with REAL test samples
   - Multi-layer detection visualization
   - Threat level alerts
   - Separate normal and attack traffic

2. **📁 CSV Upload & Analysis**
   - Auto-detection by feature count:
     - Network: 40 features
     - IoT: 5 features
     - Linux: 12 features
     - Windows: 52 features
   - Batch detection with detailed reports

3. **🧪 Manual Testing**
   - Test individual samples
   - Feature-by-feature analysis
   - Detailed error breakdown

---

## 🔍 Monitoring

### Check Container Health

```bash
# Using docker-compose
docker-compose ps

# Using docker
docker ps
docker inspect zero-day-iot-ids | grep Health
```

### View Logs

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### Resource Usage

```bash
docker stats zero-day-iot-ids
```

---

## 🛠️ Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs

# Rebuild without cache
docker-compose build --no-cache
docker-compose up -d
```

### Port 8501 already in use

```bash
# Find process using port 8501
netstat -ano | findstr :8501  # Windows
lsof -i :8501                  # Linux/Mac

# Kill the process or change port in docker-compose.yml
ports:
  - "8502:8501"  # Change 8502 to any available port
```

### Models not loading

```bash
# Verify models directory exists
ls -R models/unsupervised/

# Expected files:
# - network_autoencoder.h5
# - iot_autoencoder.h5
# - linux_autoencoder.h5
# - windows_autoencoder.h5
# - *_scaler.pkl
# - *_threshold.pkl
# - *_samples_X.npy
# - *_samples_y.npy
```

### Out of memory

Edit `docker-compose.yml`:

```yaml
services:
  ids-dashboard:
    # ... existing config ...
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

---

## 📈 Performance

### System Requirements

- **Minimum**: 2 CPU cores, 2GB RAM
- **Recommended**: 4 CPU cores, 4GB RAM
- **Storage**: ~500MB for image + models

### Expected Load Times

- Container build: 2-5 minutes (first time)
- Dashboard startup: 10-20 seconds
- Model loading: 5-10 seconds

---

## 🔄 Updates

### Update models after retraining

```bash
# Models are mounted as volume, just restart
docker-compose restart
```

### Update dashboard code

```bash
# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 🧹 Cleanup

### Remove container and images

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi zero-day-iot-ids

# Remove all unused images/volumes
docker system prune -a
```

---

## 📝 Notes

- Dashboard runs in **headless mode** (no browser auto-open)
- Health checks run every 30 seconds
- Container restarts automatically unless stopped manually
- Logs are persistent in `./logs/` directory

---

## ✅ Verification

After deployment, verify the system:

1. **Health Check**: `curl http://localhost:8501/_stcore/health`
2. **Dashboard Access**: Open browser to http://localhost:8501
3. **Model Loading**: Check dashboard sidebar shows all 4 layers loaded
4. **Detection Test**: Try Real-time mode or upload a test CSV

---

**🎉 Happy Detecting! 🎉**

Generated: 2025-12-10
