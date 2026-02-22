import torch
from PIL import Image
from torchvision import transforms
from src.model import create_retfound_model
import argparse

def run_retisense_analysis(image_path, model_path="checkpoints/final_retisense_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the model architecture
    # Allow argparse.Namespace for RETFound weights
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = create_retfound_model("models/RETFound_mae_natureCFP.pth")
    
    # 2. Load your trained weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # 3. Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Predict
    classes = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor))
        probabilities = output.cpu().numpy()[0]

    # 5. Create Report
    results = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    primary = max(results, key=results.get)
    
    # Systemic logic for the Sphinx Prize
    systemic_warning = results['Diabetes'] > 0.4 or results['Hypertension'] > 0.4

    return {
        "status": "Success",
        "predictions": results,
        "primary_diagnosis": primary,
        "systemic_alert": systemic_warning,
        "message": "Consult a GP for cardiovascular screening" if systemic_warning else "No immediate systemic risk detected"
    }