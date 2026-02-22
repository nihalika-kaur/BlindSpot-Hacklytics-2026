import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("🔍 Models available for your API Key:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        # Prints out the EXACT string you need to put in app.py
        print(m.name)