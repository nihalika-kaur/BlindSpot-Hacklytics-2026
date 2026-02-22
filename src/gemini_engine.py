import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_medical_summary(analysis_results, user_type="Patient"):
    """
    Translates model percentages into a structured medical summary.
    analysis_results: dict containing {'predictions': {disease: percentage}, 'systemic_alert': bool}
    """
    
    # Create a string version of the results for the prompt
    results_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in analysis_results['predictions'].items()])
    
    # Persona-specific instructions
    if user_type == "Patient":
        system_prompt = (
            "You are a helpful and empathetic medical assistant. "
            "Explain these eye scan results to a patient in simple, non-technical English. "
            "Use analogies where helpful (e.g., comparing a cataract to a cloudy window). "
            "If a systemic alert like Hypertension is present, explain that the eye's blood vessels "
            "can reflect heart health. Always end with a disclaimer to consult a human doctor."
        )
    else: # Doctor/Medical Pro
        system_prompt = (
            "You are a senior ophthalmology consultant. Provide a concise clinical summary for a peer. "
            "Use medical terminology (e.g., 'pathological myopia', 'hypertensive retinopathy markers'). "
            "Highlight the most significant probabilities and suggest differential diagnoses or "
            "specific follow-up screenings (like blood pressure monitoring or A1C tests)."
        )

    full_prompt = f"{system_prompt}\n\nAI Diagnostic Results: {results_str}\nSummary:"

    response = model.generate_content(full_prompt)
    return response.text

def get_chatbot_response(user_query, analysis_results, chat_history):
    """
    Handles follow-up questions using the results and history as context.
    """
    context = f"The user's previous scan results were: {analysis_results}. "
    
    # Start a chat session with history
    chat = model.start_chat(history=chat_history)
    
    response = chat.send_message(f"{context}\n\nUser Question: {user_query}")
    return response.text