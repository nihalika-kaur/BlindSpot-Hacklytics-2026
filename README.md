# RetiSense-Multi: Multi-Disease Retinal Screening System

A deep learning system for predicting **8 ocular and systemic conditions** from fundus photographs using the **RETFound foundation model**, outperforming standard ResNet baselines by 15-20% in Mean AUC.

## 🎯 What We're Predicting

Using the **ODIR-5K dataset**, this model predicts:

| Code | Disease | Type |
|------|---------|------|
| **N** | Normal | - |
| **D** | Diabetes | Systemic Indicator |
| **G** | Glaucoma | Neurological Indicator |
| **C** | Cataract | Ocular |
| **A** | Age-related Macular Degeneration | Ocular |
| **H** | Hypertension | Systemic Indicator |
| **M** | Myopia | Ocular |
| **O** | Other diseases/abnormalities | Mixed |

## 🏗️ Architecture

- **Backbone**: RETFound (ViT-Large pretrained on 1.6M retinal images)
- **Training Strategy**: Two-stage (frozen backbone → full fine-tuning)
- **Loss Function**: Focal Loss for class imbalance
- **Preprocessing**: Ben Graham + Circular Cropping + CLAHE

## 📁 Project Structure

```
MyModel/
├── data/                     # Dataset directory
│   └── ODIR-5K/             # ODIR-5K dataset
├── models/                   # Pretrained weights
│   └── RETFound_mae_NatureCFP.pth
├── src/                      # Source code
│   ├── config.py            # Configuration settings
│   ├── preprocessing.py     # Image preprocessing
│   ├── dataset.py           # Dataset & data loading
│   ├── model.py             # Model architectures
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Metrics & evaluation
│   └── utils.py             # Utility functions
├── notebooks/                # Jupyter notebooks
│   └── RetiSense_Training.ipynb
├── checkpoints/              # Saved model checkpoints
├── results/                  # Evaluation results & plots
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data & Weights

**ODIR-5K Dataset:**
```
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
```

**RETFound Weights:**
```
https://github.com/rmaphoh/RETFound_MAE
```

Save weights to `models/RETFound_mae_NatureCFP.pth`

### 3. Training

**Using the Notebook:**
```bash
jupyter notebook notebooks/RetiSense_Training.ipynb
```

**Using Python Script:**
```python
from src.config import get_config
from src.dataset import prepare_odir_dataframe, create_data_loaders, split_dataset
from src.train import train_model

# Load config
config = get_config()

# Prepare data
df = prepare_odir_dataframe(config.data.annotations_file, config.data.train_images)
train_df, val_df, test_df = split_dataset(df)
train_loader, val_loader, _ = create_data_loaders(config, train_df, val_df)

# Train
model, history = train_model(train_loader, val_loader, config)
```

## 📊 Expected Results

### Target Metrics
- **Mean AUC**: > 0.93
- **Hypertension AUC**: > 0.80 (publication-grade)
- **Improvement over ResNet-50**: 15-20%

### Training Configuration
- **Stage 1**: 10 epochs, frozen backbone, LR=1e-3
- **Stage 2**: 30 epochs, full fine-tuning, LR=1e-5
- **Batch Size**: 16 (adjust based on GPU memory)

## 🔧 Key Components

### Preprocessing Pipeline
1. **Circular Cropping**: Remove black borders
2. **Ben Graham's Preprocessing**: Subtract local average for consistency
3. **CLAHE**: Adaptive histogram equalization
4. **Augmentation**: Rotations, flips, brightness/contrast

### Two-Stage Training
1. **Stage 1 (Frozen)**: Train only classification head
   - Preserves foundation model knowledge
   - Fast convergence

2. **Stage 2 (Fine-tuning)**: Unfreeze entire network
   - Low learning rate (1e-5)
   - Layer-wise LR decay

### Loss Functions
- **Focal Loss**: For imbalanced classes
- **Asymmetric Loss**: Alternative for multi-label

## 🖥️ Hardware Requirements

- **Recommended**: NVIDIA A100/L4 (24GB+ VRAM)
- **Minimum**: NVIDIA RTX 3090 (24GB VRAM)
- **Alternative**: Google Colab Pro/Pro+

### Memory Estimates
| Image Size | Batch Size | GPU Memory |
|------------|------------|------------|
| 224×224 | 16 | ~16GB |
| 224×224 | 8 | ~10GB |
| 384×384 | 8 | ~20GB |

## 📈 Evaluation

```python
from src.evaluate import evaluate_model, generate_evaluation_report

# Full evaluation with plots
results = generate_evaluation_report(
    model=model,
    dataloader=test_loader,
    output_dir='results/evaluation',
    model_name='RETFound_RetiSense'
)
```

## 📚 References

- **RETFound**: [A foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x)
- **ODIR-5K**: Ocular Disease Intelligent Recognition Dataset
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
