# Dataset Directory

This directory contains the TON-IoT datasets for training and testing the Zero-Day IDS.

## 📁 Directory Structure

```
data/
├── Train_Test_datasets/
│   ├── Train_Test_Network_dataset/
│   │   └── train_test_network.csv (211K samples, ~40 features)
│   ├── Train_Test_IoT_dataset/
│   │   └── Train_Test_IoT_Modbus.csv (31K samples, ~5 features)
│   ├── Train_Test_Linux_dataset/
│   │   └── Train_Test_Linux_process.csv (90K samples, ~12 features)
│   └── Train_Test_Windows_dataset/
│       └── Train_Test_Windows_10.csv (21K samples, ~52 features)
└── README.md (this file)
```

## 📥 Downloading the Dataset

The datasets are from the **TON-IoT Dataset** by UNSW Canberra Cyber.

**Download Links:**
- Official: https://cloudstor.aarnet.edu.au/plus/s/ds5zW91vdgjEj9i
- Alternative: https://research.unsw.edu.au/projects/toniot-datasets

## 📋 Required Files

After downloading, place these CSV files in their respective directories:

| Layer | File Path | Size | Samples |
|-------|-----------|------|---------|
| **Network** | `Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv` | ~500 MB | 211,043 |
| **IoT** | `Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Modbus.csv` | ~10 MB | 31,106 |
| **Linux** | `Train_Test_datasets/Train_Test_Linux_dataset/Train_Test_Linux_process.csv` | ~50 MB | 90,112 |
| **Windows** | `Train_Test_datasets/Train_Test_Windows_dataset/Train_Test_Windows_10.csv` | ~200 MB | 21,104 |

**Total size:** ~760 MB

## ⚠️ Important Note

**CSV files are NOT included in the Git repository** due to their large size.

You MUST download them separately from the official TON-IoT dataset source.

## 🏷️ Dataset Labels

All datasets use the `type` column for labels:
- `normal` - Normal traffic/behavior
- Attack types: `backdoor`, `ddos`, `dos`, `injection`, `password`, `scanning`, `ransomware`, `xss`, `mitm`

## 📚 Citation

If you use the TON-IoT dataset in your research, please cite:

```bibtex
@article{moustafa2020toniot,
  title={A new distributed architecture for evaluating AI-based security systems at the edge: Network TON\_IoT datasets},
  author={Moustafa, Nour and Ahmed, Marwa and Ahmed, Shaad},
  journal={Sustainable Cities and Society},
  year={2020},
  publisher={Elsevier}
}
```

## 🚀 Quick Setup

1. Download datasets from the link above
2. Extract and place CSV files in the correct directories
3. Run training: `cd src && python train_unsupervised.py`

The training script will automatically load and preprocess the datasets.
