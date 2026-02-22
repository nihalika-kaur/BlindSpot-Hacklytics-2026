# import os
# import sys
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# import argparse

# # Ensure the project root is in the path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.dataset import ODIRDataset, prepare_odir_df
# from src.model import create_retfound_model

# def run_fast_fine_tune():
#     # 1. CONFIGURATION
#     IMG_DIR = r"data\ODIR-5K\ODIR-5K\Training Images"
#     CSV_PATH = r"data\full_df.csv"
    
#     # FIX: Pointing to the specific file saved at the end of Stage 1
#     STAGE1_WEIGHTS = r"checkpoints\stage1_retfound.pth" 
#     BASE_WEIGHTS = r"models\RETFound_mae_natureCFP.pth"
    
#     BATCH_SIZE = 4 
#     LEARNING_RATE = 5e-5 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 2. DATA PREPARATION
#     print("📊 Loading dataset for fine-tuning...")
#     df = prepare_odir_df(CSV_PATH)
#     train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     train_loader = DataLoader(ODIRDataset(train_df, IMG_DIR, transform), batch_size=BATCH_SIZE, shuffle=True)

#     # 3. MODEL SETUP
#     # Allow loading the argparse.Namespace from RETFound
#     torch.serialization.add_safe_globals([argparse.Namespace])
    
#     model = create_retfound_model(BASE_WEIGHTS)
    
#     print(f"🔄 Loading Stage 1 weights from {STAGE1_WEIGHTS}...")
#     # FIX: Added weights_only=False to prevent the security warning/crash
#     model.load_state_dict(torch.load(STAGE1_WEIGHTS, map_location=device, weights_only=False))
    
#     # 4. PARTIAL UNFREEZING (The Speed Hack)
#     # Freeze everything first
#     for param in model.parameters():
#         param.requires_grad = False
    
#     # Unfreeze only the LAST transformer block and the head
#     # This keeps the medical foundation intact but specializes the high-level features
#     for param in model.blocks[-1].parameters():
#         param.requires_grad = True
#     for param in model.norm.parameters():
#         param.requires_grad = True
#     for param in model.head.parameters():
#         param.requires_grad = True

#     model.to(device)
    
#     # 5. OPTIMIZER & LOSS
#     # filter() ensures we only send the unfrozen parameters to the optimizer
#     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
#     criterion = nn.BCEWithLogitsLoss()

#     # 6. TRAINING LOOP (1 EPOCH)
#     print(f"🔥 Starting Lightning Stage 2 (Last Block Only) on {device}...")
#     model.train()
    
#     for batch_idx, (imgs, lbls) in enumerate(train_loader):
#         imgs, lbls = imgs.to(device), lbls.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, lbls)
#         loss.backward()
#         optimizer.step()
        
#         if batch_idx % 50 == 0:
#             print(f"Fine-tune | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

#     # 7. SAVE FINAL MODEL
#     os.makedirs("checkpoints", exist_ok=True)
#     torch.save(model.state_dict(), "checkpoints/final_retisense_model.pth")
#     print("🏁 FINAL MODEL SAVED: checkpoints/final_retisense_model.pth")

# if __name__ == "__main__":
#     run_fast_fine_tune()

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import argparse

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import ODIRDataset, prepare_odir_df
from src.model import create_retfound_model

def run_extended_fine_tune():
    # 1. CONFIGURATION
    IMG_DIR = r"data\ODIR-5K\ODIR-5K\Training Images"
    CSV_PATH = r"data\full_df.csv"
    
    # We load the weights from your first successful 1-epoch run
    PREVIOUS_WEIGHTS = r"checkpoints\final_retisense_model.pth" 
    BASE_MODEL_PATH = r"models\RETFound_mae_natureCFP.pth"
    
    BATCH_SIZE = 4 
    # Lowered slightly for the final push to avoid "over-stepping"
    LEARNING_RATE = 3e-5 
    EPOCHS = 2 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. DATA PREPARATION
    print("📊 Preparing data for the final 2 epochs...")
    df = prepare_odir_df(CSV_PATH)
    train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(ODIRDataset(train_df, IMG_DIR, transform), batch_size=BATCH_SIZE, shuffle=True)

    # 3. MODEL SETUP
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = create_retfound_model(BASE_MODEL_PATH)
    
    print(f"🔄 Resuming from previous fine-tuned weights: {PREVIOUS_WEIGHTS}")
    model.load_state_dict(torch.load(PREVIOUS_WEIGHTS, map_location=device, weights_only=False))
    
    # 4. SELECTIVE UNFREEZING
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreezing the last block and head again
    for param in model.blocks[-1].parameters():
        param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    model.to(device)
    
    # 5. OPTIMIZER & SCHEDULER
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    # This reduces LR by 50% after the first epoch to fine-tune the final details
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    # 6. TRAINING LOOP
    print(f"🚀 Starting the final {EPOCHS} epochs of refinement...")
    
    for epoch in range(EPOCHS):
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
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

    # 7. SAVE FINAL VERSION
    os.makedirs("checkpoints", exist_ok=True)
    final_path = "checkpoints/retisense_final_v2.pth"
    torch.save(model.state_dict(), final_path)
    print(f"🏁 MISSION COMPLETE: {final_path}")

if __name__ == "__main__":
    run_extended_fine_tune()