import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from src.dataset import ODIRDataset, prepare_odir_df
from src.model import create_retfound_model
from torchvision import transforms
from sklearn.model_selection import train_test_split
import argparse

def run_comparison():
    device = torch.device("cuda")
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    # 1. Setup Data
    df = prepare_odir_df(r"data\full_df.csv")
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = ODIRDataset(test_df, r"data\ODIR-5K\ODIR-5K\Training Images", transform)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    # 2. Paths
    v1_path = "checkpoints/final_retisense_model.pth"
    v2_path = "checkpoints/retisense_final_v2.pth"
    base_model = "models/RETFound_mae_natureCFP.pth"

    def get_scores(path):
        model = create_retfound_model(base_model)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.to(device).eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                outputs = torch.sigmoid(model(imgs.to(device)))
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(lbls.numpy())
        return np.vstack(all_preds), np.vstack(all_labels)

    print("📊 Evaluating V1 (1-Epoch)...")
    preds1, labels = get_scores(v1_path)
    auc1 = roc_auc_score(labels, preds1, average='macro')

    print("📊 Evaluating V2 (3-Epoch)...")
    preds2, _ = get_scores(v2_path)
    auc2 = roc_auc_score(labels, preds2, average='macro')

    print("\n" + "="*30)
    print(f"V1 Macro AUC: {auc1:.4f}")
    print(f"V2 Macro AUC: {auc2:.4f}")
    print("="*30)
    
    if auc2 > auc1:
        print(f"✅ V2 is better by {((auc2-auc1)/auc1):.2%}. Use retisense_final_v2.pth!")
    else:
        print("⚠️ V1 is actually slightly better (Overfitting occurred). Stick with final_retisense_model.pth.")

if __name__ == "__main__":
    run_comparison()