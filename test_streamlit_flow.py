"""Test the exact flow that Streamlit uses"""
import tempfile
from PIL import Image
from inference import run_retisense_analysis

# Same image as test_demo.py
test_image = r'data\ODIR-5K\ODIR-5K\Testing Images\937_left.jpg'

print("1. Opening image with PIL (like Streamlit does)...")
image = Image.open(test_image)
print(f"   Image size: {image.size}, mode: {image.mode}")

print("\n2. Saving to temp file (like Streamlit does)...")
with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
    image.save(f.name)
    temp_path = f.name
print(f"   Temp file: {temp_path}")

print("\n3. Running inference on temp file...")
report = run_retisense_analysis(temp_path, model_path='checkpoints/retisense_final_v2.pth')

print("\n" + "="*40)
print("STREAMLIT FLOW TEST RESULTS")
print("="*40)
print(f"Primary Diagnosis: {report['primary_diagnosis']}")
print(f"Systemic Alert: {report['systemic_alert']}")
print(f"Message: {report['message']}")
print("-"*40)
print("Predictions:")
for k, v in report['predictions'].items():
    print(f"  {k}: {v:.2%}")
print("="*40)
