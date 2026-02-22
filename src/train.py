# import os
# import sys
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from sklearn.model_selection import train_test_split

# # FIX: Allow Python to see the 'src' folder from the root
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.dataset import ODIRDataset, prepare_odir_df
# from src.model import create_retfound_model

# def run_training():
#     # 1. Config
#     # Change this to match your actual folder name
#     IMG_DIR = "data/ODIR-5K/ODIR-5K/Training Images"
#     CSV_PATH = "data/full_df.csv"
#     WEIGHTS_PATH = "models/RETFound_mae_natureCFP.pth"
#     BATCH_SIZE = 4  # Small batch size for ViT-Large memory limits
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs("checkpoints", exist_ok=True)

#     # 2. Data
#     df = prepare_odir_df(CSV_PATH)
#     train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     train_loader = DataLoader(ODIRDataset(train_df, IMG_DIR, transform), batch_size=BATCH_SIZE, shuffle=True)
    
#     # 3. Model Stage 1: Freeze Backbone
#     model = create_retfound_model(WEIGHTS_PATH)
#     for param in model.parameters():
#         param.requires_grad = False
#     for param in model.head.parameters():
#         param.requires_grad = True
        
#     model.to(device)
#     optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
#     criterion = nn.BCEWithLogitsLoss()

#     print(f"Starting Stage 1 Training on {device}...")
#     for epoch in range(5):
#         model.train()
#         epoch_loss = 0
#         for imgs, lbls in train_loader:
#             imgs, lbls = imgs.to(device), lbls.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(imgs), lbls)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f}")

#     torch.save(model.state_dict(), "checkpoints/stage1_retfound.pth")
#     print("Stage 1 Complete!")

# if __name__ == "__main__":
#     run_training()

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# FIX: Allow Python to see the 'src' folder from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import ODIRDataset, prepare_odir_df
from src.model import create_retfound_model

def run_training():
    # 1. Config - USING RAW STRING FOR WINDOWS PATH
    IMG_DIR = r"data\ODIR-5K\ODIR-5K\Training Images"
    CSV_PATH = r"data\full_df.csv"
    WEIGHTS_PATH = r"models\RETFound_mae_natureCFP.pth"
    
    # ADJUST BATCH SIZE: 4 for GPU, 1-2 if you get 'Out of Memory'
    BATCH_SIZE = 4  
    
    # Automatically detect GPU (NVIDIA), Mac (MPS), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("⚠️ WARNING: No GPU detected. Training will be extremely slow!")

    os.makedirs("checkpoints", exist_ok=True)

    # 2. Data
    print("📊 Loading dataset...")
    df = prepare_odir_df(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ODIRDataset(train_df, IMG_DIR, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 3. Model Stage 1: Freeze Backbone
    model = create_retfound_model(WEIGHTS_PATH)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
        
    model.to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print(f"🚀 Starting Stage 1 Training on {device}...")
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Complete | Average Loss: {epoch_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/stage1_epoch_{epoch+1}.pth")

    print("🏁 Stage 1 Complete! Weights saved to checkpoints folder.")

if __name__ == "__main__":
    run_training()