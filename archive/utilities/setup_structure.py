"""
Script để tạo cấu trúc thư mục project
Chạy script này TRƯỚC khi train
"""

import os

def setup_project_structure():
    """Tạo cấu trúc thư mục chuẩn"""
    
    directories = [
        'data',                    # Chứa dataset và test data
        'models',                  # Chứa trained model, scaler, threshold
        'results',                 # Chứa ảnh kết quả, reports
        'src',                     # Chứa source code
        'notebooks',               # Jupyter notebooks (optional)
        'docs',                    # Documentation, papers
    ]
    
    print("="*60)
    print("SETTING UP PROJECT STRUCTURE")
    print("="*60)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[✓] Created/verified: {directory}/")
    
    # Create README files
    readme_content = {
        'data/README.md': """# Data Directory

Place your TON_IoT dataset files here.

Recommended structure:
```
data/
├── Train_Test_datasets/
│   ├── Network_dataset_train.csv
│   └── Network_dataset_test.csv
├── test_data.npy          (generated during training)
├── test_labels.npy        (generated during training)
└── test_zero_day.npy      (generated during training)
```
""",
        'models/README.md': """# Models Directory

Trained models and preprocessing objects are saved here.

Files generated:
- `ton_iot_autoencoder.h5` - Trained autoencoder model
- `scaler.pkl` - StandardScaler for preprocessing
- `threshold.pkl` - Optimal anomaly threshold
""",
        'results/README.md': """# Results Directory

Training results, visualizations, and reports are saved here.

Generated files:
- `ton_iot_results.png` - Training results and ROC curve
- `detection_results.png` - Detection performance (from demo)
- `severity_analysis.png` - Attack severity analysis (from demo)
""",
        'src/README.md': """# Source Code Directory

Main source code files:
- `train.py` - Training script
- `inference.py` - Demo and inference script
- `utils.py` - Helper functions (optional)
"""
    }
    
    for filepath, content in readme_content.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[✓] Created: {filepath}")
    
    print("\n" + "="*60)
    print("PROJECT STRUCTURE SETUP COMPLETE!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Place your TON_IoT dataset in: data/Train_Test_datasets/")
    print("2. Run: python src/train.py")
    print("3. Run: python src/inference.py")
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data (don't commit large files)
data/*.csv
data/*.npy
*.csv
*.npy

# Models (don't commit large model files)
models/*.h5
models/*.keras
models/*.pkl

# Results
results/*.png
results/*.jpg
results/*.pdf

# OS
.DS_Store
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("[✓] Created: .gitignore")
    
    print("\n✨ Setup complete! Your project is ready.")

if __name__ == "__main__":
    setup_project_structure()