import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from src.dataset import ODIRDataset, prepare_odir_df
from src.model import create_retfound_model
from torchvision import transforms
from sklearn.model_selection import train_test_split
import argparse

def evaluate_retisense():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([argparse.Namespace])

    # 1. Setup Data
    print("📊 Preparing test data...")
    df = prepare_odir_df(r"data\full_df.csv")
    _, test_df = train_test_split(df, test_size=0.1, random_state=42) # Same split as training
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = ODIRDataset(test_df, r"data\ODIR-5K\ODIR-5K\Training Images", transform)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    # 2. Load Model
    print("🧠 Loading final model...")
    model = create_retfound_model("models/RETFound_mae_natureCFP.pth")
    model.load_state_dict(torch.load("checkpoints/final_retisense_model.pth", map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # 3. Inference
    all_preds = []
    all_labels = []
    
    print(f"🔎 Evaluating {len(test_df)} images...")
    with torch.no_grad():
        for imgs, lbls in loader:
            outputs = torch.sigmoid(model(imgs.to(device)))
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(lbls.numpy())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    # 4. Calculate Metrics
    classes = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    
    print("\n" + "="*40)
    print("       MODEL PERFORMANCE REPORT       ")
    print("="*40)
    
    overall_auc = roc_auc_score(labels, preds, average='macro')
    print(f"OVERALL MACRO AUC: {overall_auc:.4f}")
    print("-" * 40)
    
    for i, class_name in enumerate(classes):
        auc = roc_auc_score(labels[:, i], preds[:, i])
        # Calculate accuracy at 0.5 threshold
        acc = accuracy_score(labels[:, i], (preds[:, i] > 0.5).astype(int))
        print(f"{class_name:12} | AUC: {auc:.4f} | Acc: {acc:.2%}")
    
    print("="*40)

if __name__ == "__main__":
    evaluate_retisense()