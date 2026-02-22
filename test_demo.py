import torch
from inference import run_retisense_analysis
import json
import os

# 1. Path to an image you want to test (check your folder for a real filename)
# Try picking '0_left.jpg' or any other image in your training folder
# test_image = r"data\ODIR-5K\ODIR-5K\Training Images\937_left.jpg" 
test_image = r"data\ODIR-5K\ODIR-5K\Testing Images\937_left.jpg" 
if not os.path.exists(test_image):
    print(f"❌ Error: Could not find image at {test_image}")
else:
    print("🧠 RetiSense is thinking...")
    
    # 2. Run analysis using your NEWLY saved final weights
    report = run_retisense_analysis(test_image, model_path="checkpoints/final_retisense_model.pth")

    print("\n" + "="*30)
    print("      RETISENSE REPORT      ")
    print("="*30)
    print(f"Primary Diagnosis: {report['primary_diagnosis']}")
    print(f"Systemic Alert:    {'⚠️ YES' if report['systemic_alert'] else '✅ NO'}")
    print(f"Clinical Note:     {report['message']}")
    print("-" * 30)
    print("Confidence Breakdown:")
    for disease, prob in report['predictions'].items():
        print(f"  {disease:12}: {prob:.2%}")
    print("="*30)