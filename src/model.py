import torch
import torch.nn as nn
import timm
import argparse # Needed to allowlist the Namespace object

def create_retfound_model(checkpoint_path, num_classes=8):
    # FIX: Allow PyTorch 2.6+ to load the Namespace object found in RETFound weights
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    # Create the ViT-Large backbone used by RETFound
    model = timm.create_model('vit_large_patch16_224', pretrained=False)
    
    print(f"Loading RETFound weights from {checkpoint_path}...")
    
    # Use weights_only=False to bypass the strict pickling check
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict (RETFound usually wraps it in a 'model' key)
    state_dict = checkpoint.get('model', checkpoint)
    
    # Load weights into the backbone
    # strict=False allows us to ignore the original pretrained head
    msg = model.load_state_dict(state_dict, strict=False)
    
    # Replace the classification head for your 8 disease categories
    model.head = nn.Linear(model.embed_dim, num_classes)
    
    return model