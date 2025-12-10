# Source Code Directory

## Main Scripts
- [train.py](train.py) - Main training pipeline orchestrator
- [inference.py](inference.py) - Batch inference and demo modes
- [realtime_simple.py](realtime_simple.py) - Real-time detection system (recommended)
- [realtime_detector.py](realtime_detector.py) - Production template with feature extraction

## Training Modules (Used by train.py)
- [data_loader.py](data_loader.py) - Dataset loading and exploration
- [preprocessor.py](preprocessor.py) - Data preprocessing and zero-day split
- [model_builder.py](model_builder.py) - Autoencoder architecture and training
- [threshold_finder.py](threshold_finder.py) - Threshold optimization (5 methods)
- [evaluator.py](evaluator.py) - Model evaluation and visualization

## Usage

### Training
```bash
cd src
python train.py
```

### Inference
```bash
cd src
python inference.py
```

### Real-time Detection
```bash
# Windows
run_realtime.bat

# Linux/Mac
cd src
python realtime_simple.py
```
