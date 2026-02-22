# """
# BlindSpot: AI-Powered Retinal Disease Screening
# ======================================================
# A Streamlit application for multi-disease detection from fundus images.
# """

# import streamlit as st
# import torch
# import numpy as np
# from PIL import Image
# import time
# import os
# import tempfile
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Import the actual inference function from inference.py
# from inference import run_retisense_analysis

# # Load environment variables
# load_dotenv()

# # Page configuration - MUST be first Streamlit command
# st.set_page_config(
#     page_title="BlindSpot | AI Retinal Screening",
#     page_icon="👁️",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # ============== CONSTANTS ==============
# DISEASE_LABELS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
# DISEASE_FULL_NAMES = {
#     'Normal': 'Normal/Healthy',
#     'Diabetes': 'Diabetic Retinopathy',
#     'Glaucoma': 'Glaucoma',
#     'Cataract': 'Cataract',
#     'AMD': 'Age-related Macular Degeneration',
#     'Hypertension': 'Hypertensive Retinopathy',
#     'Myopia': 'Pathological Myopia',
#     'Other': 'Other Abnormalities'
# }
# DISEASE_COLORS = {
#     'Normal': '#2ecc71',      # Green
#     'Diabetes': '#e74c3c',    # Red
#     'Glaucoma': '#9b59b6',    # Purple
#     'Cataract': '#3498db',    # Blue
#     'AMD': '#f39c12',         # Orange
#     'Hypertension': '#e91e63', # Pink
#     'Myopia': '#00bcd4',      # Cyan
#     'Other': '#95a5a6'        # Gray
# }
# SYSTEMIC_DISEASES = ['Diabetes', 'Hypertension']

# # Model path - same as used in test_demo.py (your trained model)
# MODEL_PATH = "checkpoints/retisense_final_v2.pth"


# # ============== CUSTOM CSS ==============
# def load_css():
#     st.markdown("""
#     <style>
#     /* Main container styling */
#     .main-header {
#         text-align: center;
#         padding: 2rem 0;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 20px;
#         margin-bottom: 2rem;
#         color: white;
#     }
    
#     .main-header h1 {
#         font-size: 3rem;
#         margin-bottom: 0.5rem;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
#     }
    
#     .main-header p {
#         font-size: 1.2rem;
#         opacity: 0.9;
#     }
    
#     /* Card styling */
#     .card {
#         background: white;
#         border-radius: 15px;
#         padding: 2rem;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#         margin-bottom: 1rem;
#         border: 1px solid #e0e0e0;
#     }
    
#     .card-dark {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#         color: white;
#     }
    
#     /* User type selection cards */
#     .user-card {
#         background: white;
#         border-radius: 20px;
#         padding: 2rem;
#         text-align: center;
#         cursor: pointer;
#         transition: all 0.3s ease;
#         border: 3px solid transparent;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#     }
    
#     .user-card:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 8px 25px rgba(0,0,0,0.15);
#     }
    
#     .user-card.selected {
#         border-color: #667eea;
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#     }
    
#     .user-card-icon {
#         font-size: 4rem;
#         margin-bottom: 1rem;
#     }
    
#     /* Result bars */
#     .result-bar-container {
#         margin: 1rem 0;
#         background: #f5f5f5;
#         border-radius: 10px;
#         padding: 1rem;
#     }
    
#     .result-bar-label {
#         display: flex;
#         justify-content: space-between;
#         margin-bottom: 0.5rem;
#         font-weight: 600;
#     }
    
#     .result-bar {
#         height: 25px;
#         border-radius: 12px;
#         transition: width 0.5s ease;
#     }
    
#     /* Alert boxes */
#     .systemic-alert {
#         background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#         display: flex;
#         align-items: center;
#         gap: 1rem;
#     }
    
#     .systemic-alert-icon {
#         font-size: 2rem;
#     }
    
#     /* Summary box */
#     .summary-box {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 15px;
#         margin-top: 2rem;
#         line-height: 1.8;
#     }
    
#     .summary-box h3 {
#         margin-bottom: 1rem;
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#     }
    
#     /* Chatbot styling */
#     .chat-message {
#         padding: 1rem;
#         border-radius: 15px;
#         margin: 0.5rem 0;
#         max-width: 85%;
#     }
    
#     .chat-user {
#         background: #667eea;
#         color: white;
#         margin-left: auto;
#     }
    
#     .chat-assistant {
#         background: #f0f2f6;
#         color: #333;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.75rem 2rem;
#         border-radius: 25px;
#         font-size: 1.1rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         transform: scale(1.05);
#         box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
#     }
    
#     /* Upload area */
#     .upload-area {
#         border: 3px dashed #667eea;
#         border-radius: 20px;
#         padding: 3rem;
#         text-align: center;
#         background: #f8f9ff;
#         margin: 2rem 0;
#     }
    
#     /* Progress styling */
#     .stProgress > div > div {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     /* Sidebar styling */
#     .css-1d391kg {
#         background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
#     }
    
#     </style>
#     """, unsafe_allow_html=True)


# # ============== RUN INFERENCE (Using actual inference.py) ==============
# def run_analysis(image_path: str) -> dict:
#     """
#     Run the actual RetiSense model inference.
#     Uses the same pipeline as test_demo.py with the trained model.
#     """
#     return run_retisense_analysis(image_path, model_path=MODEL_PATH)


# # ============== GEMINI SETUP ==============
# def setup_gemini():
#     """Configure Gemini API."""
#     api_key = os.getenv("GEMINI_API_KEY")
#     if api_key:
#         genai.configure(api_key=api_key)
#         return genai.GenerativeModel('gemini-1.5-flash')
#     return None


# # ============== GEMINI SUMMARY GENERATION ==============
# def generate_summary(predictions, user_type, gemini_model, systemic_alert=False, primary_diagnosis="Unknown"):
#     """Generate AI summary based on predictions and user type."""
#     if gemini_model is None:
#         return "⚠️ Gemini API not configured. Please add GEMINI_API_KEY to .env file."
    
#     results_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in predictions.items()])
    
#     if user_type == "Patient":
#         prompt = f"""You are a caring and empathetic medical assistant explaining eye scan results to a patient.

# The AI analysis of the retinal scan shows:
# {results_str}

# Primary Finding: {primary_diagnosis}
# {"⚠️ IMPORTANT: A systemic health alert was triggered. This may indicate underlying conditions that affect the whole body, not just the eyes." if systemic_alert else ""}

# Please:
# 1. Explain what the primary finding ({primary_diagnosis}) means in simple, everyday language
# 2. Use helpful analogies (e.g., comparing a cataract to looking through a foggy window)
# 3. If systemic indicators (Diabetes/Hypertension) are elevated, gently explain that eye blood vessels can reflect overall health
# 4. Reassure the patient while encouraging them to follow up with their doctor
# 5. Keep it warm, supportive, and easy to understand
# 6. End with a brief disclaimer that this is AI-assisted and requires professional medical confirmation

# Keep the response concise but thorough (about 150-200 words)."""
#     else:
#         prompt = f"""You are a senior ophthalmology consultant providing a clinical summary to a fellow healthcare professional.

# AI Diagnostic Screening Results:
# {results_str}

# Primary Finding: {primary_diagnosis}
# {"⚠️ SYSTEMIC ALERT: Elevated probability for systemic indicators detected." if systemic_alert else ""}

# Please provide:
# 1. A concise clinical interpretation using appropriate medical terminology
# 2. Highlight the most clinically significant findings (probabilities > 30%)
# 3. For systemic indicators, note relevant follow-up recommendations (e.g., HbA1c, BP monitoring)
# 4. Suggest differential diagnoses if applicable
# 5. Recommend appropriate referrals or additional imaging (OCT, visual fields, etc.)
# 6. Use abbreviations and terminology appropriate for medical documentation

# Format as a brief clinical note (about 150-200 words)."""

#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error generating summary: {e}"


# # ============== CHATBOT RESPONSE ==============
# def get_chatbot_response(query, analysis_report, chat_history, user_type, gemini_model):
#     """Generate chatbot response for follow-up questions."""
#     if gemini_model is None:
#         return "⚠️ Chatbot requires Gemini API. Please configure GEMINI_API_KEY."
    
#     predictions = analysis_report.get('predictions', {})
#     results_context = ", ".join([f"{k}: {v*100:.1f}%" for k, v in predictions.items()])
    
#     persona = "empathetic medical assistant using simple language" if user_type == "Patient" else "clinical ophthalmology consultant using medical terminology"
    
#     # Build conversation context
#     history_text = "\n".join([f"{'User' if i%2==0 else 'Assistant'}: {msg}" for i, msg in enumerate(chat_history[-6:])])
    
#     prompt = f"""You are a {persona} helping with follow-up questions about retinal scan results.

# Scan Analysis:
# - Predictions: {results_context}
# - Primary Diagnosis: {analysis_report.get('primary_diagnosis', 'Unknown')}
# - Systemic Alert: {'Yes' if analysis_report.get('systemic_alert', False) else 'No'}
# - Clinical Note: {analysis_report.get('message', '')}

# Recent conversation:
# {history_text}

# User's new question: {query}

# Provide a helpful, accurate response. {"Use simple language and analogies." if user_type == "Patient" else "Use appropriate medical terminology."} 
# If asked about treatment, always recommend consulting with their healthcare provider.
# Keep responses concise (2-3 paragraphs max)."""
    
#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error: {e}"


# # ============== UI COMPONENTS ==============
# def render_home_page():
#     """Render the landing/home page."""
#     st.markdown("""
#     <div class="main-header">
#         <h1>👁️ BlindSpot</h1>
#         <p>AI-Powered Multi-Disease Retinal Screening System</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col2:
#         st.markdown("""
#         <div class="card" style="text-align: center;">
#             <h2>🔬 Advanced AI Screening</h2>
#             <p style="font-size: 1.1rem; color: #666; margin: 1.5rem 0;">
#                 Our state-of-the-art AI model, powered by <strong>RETFound</strong> 
#                 (trained on 1.6 million retinal images), can detect <strong>8 different conditions</strong> 
#                 from a single fundus photograph.
#             </p>
            
#             <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin: 2rem 0;">
#                 <span style="background: #e8f5e9; color: #2e7d32; padding: 0.5rem 1rem; border-radius: 20px;">✓ Diabetic Retinopathy</span>
#                 <span style="background: #f3e5f5; color: #7b1fa2; padding: 0.5rem 1rem; border-radius: 20px;">✓ Glaucoma</span>
#                 <span style="background: #e3f2fd; color: #1565c0; padding: 0.5rem 1rem; border-radius: 20px;">✓ Cataract</span>
#                 <span style="background: #fff3e0; color: #ef6c00; padding: 0.5rem 1rem; border-radius: 20px;">✓ AMD</span>
#                 <span style="background: #fce4ec; color: #c2185b; padding: 0.5rem 1rem; border-radius: 20px;">✓ Hypertension</span>
#                 <span style="background: #e0f7fa; color: #00838f; padding: 0.5rem 1rem; border-radius: 20px;">✓ Myopia</span>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown("<br>", unsafe_allow_html=True)
        
#         col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
#         with col_btn2:
#             if st.button("🚀 Try It Now", use_container_width=True, key="try_it"):
#                 st.session_state.page = "select_user"
#                 st.rerun()
        
#         st.markdown("""
#         <div style="text-align: center; margin-top: 2rem; color: #999; font-size: 0.9rem;">
#             <p>⚠️ This tool is for screening purposes only and does not replace professional medical diagnosis.</p>
#         </div>
#         """, unsafe_allow_html=True)


# def render_user_selection():
#     """Render user type selection page."""
#     st.markdown("""
#     <div class="main-header">
#         <h1>👤 Who's Using RetiSense?</h1>
#         <p>Select your profile for a personalized experience</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1, 1, 1])
    
#     with col1:
#         st.markdown("""
#         <div class="card" style="text-align: center; height: 300px;">
#             <div style="font-size: 5rem;">🧑‍🤝‍🧑</div>
#             <h3>Patient</h3>
#             <p style="color: #666;">Get easy-to-understand results explained in plain English</p>
#         </div>
#         """, unsafe_allow_html=True)
#         if st.button("I'm a Patient", use_container_width=True, key="patient_btn"):
#             st.session_state.user_type = "Patient"
#             st.session_state.page = "upload"
#             st.rerun()
    
#     with col3:
#         st.markdown("""
#         <div class="card" style="text-align: center; height: 300px;">
#             <div style="font-size: 5rem;">👨‍⚕️</div>
#             <h3>Medical Professional</h3>
#             <p style="color: #666;">Get detailed clinical analysis with medical terminology</p>
#         </div>
#         """, unsafe_allow_html=True)
#         if st.button("I'm a Doctor/Professional", use_container_width=True, key="doctor_btn"):
#             st.session_state.user_type = "Doctor"
#             st.session_state.page = "upload"
#             st.rerun()
    
#     with col2:
#         st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)
    
#     # Back button
#     st.markdown("<br>", unsafe_allow_html=True)
#     if st.button("← Back to Home"):
#         st.session_state.page = "home"
#         st.rerun()


# def render_upload_page():
#     """Render image upload page."""
#     user_type = st.session_state.get("user_type", "Patient")
    
#     st.markdown(f"""
#     <div class="main-header">
#         <h1>📤 Upload Retinal Image</h1>
#         <p>Upload a fundus photograph for AI analysis • Mode: {user_type}</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col2:
#         st.markdown("""
#         <div class="card">
#             <h3 style="text-align: center;">📷 Select Image</h3>
#             <p style="text-align: center; color: #666;">
#                 Upload a high-quality fundus photograph (JPG, PNG, JPEG)
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         uploaded_file = st.file_uploader(
#             "Choose a fundus image",
#             type=["jpg", "jpeg", "png"],
#             label_visibility="collapsed"
#         )
        
#         if uploaded_file:
#             image = Image.open(uploaded_file)
            
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.image(image, caption="Uploaded Image", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
            
#             if st.button("🔍 Analyze Image", use_container_width=True):
#                 # Save to temp file for inference
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
#                     image.save(tmp_file.name)
#                     st.session_state.temp_image_path = tmp_file.name
#                 st.session_state.uploaded_image = image
#                 st.session_state.page = "results"
#                 st.rerun()
    
#     # Back button
#     st.markdown("<br>", unsafe_allow_html=True)
#     col_back, _, _ = st.columns([1, 2, 1])
#     with col_back:
#         if st.button("← Change User Type"):
#             st.session_state.page = "select_user"
#             st.rerun()


# def render_results_page():
#     """Render results page with analysis."""
#     user_type = st.session_state.get("user_type", "Patient")
#     image = st.session_state.get("uploaded_image")
#     temp_image_path = st.session_state.get("temp_image_path")
    
#     if image is None or temp_image_path is None:
#         st.session_state.page = "upload"
#         st.rerun()
#         return
    
#     st.markdown(f"""
#     <div class="main-header">
#         <h1>📊 Analysis Results</h1>
#         <p>AI-Powered Screening Report • {user_type} View</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     gemini_model = setup_gemini()
    
#     # Show progress bar during analysis
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     status_text.text("🔄 Loading RetiSense model...")
#     progress_bar.progress(20)
    
#     status_text.text("🧠 Running AI analysis (using trained model)...")
#     progress_bar.progress(50)
    
#     # Run the ACTUAL inference using inference.py
#     try:
#         analysis_report = run_analysis(temp_image_path)
#         predictions = analysis_report['predictions']
#         primary_diagnosis = analysis_report['primary_diagnosis']
#         systemic_alert = analysis_report['systemic_alert']
#         clinical_message = analysis_report['message']
#     except Exception as e:
#         st.error(f"Error running model: {e}")
#         st.stop()
    
#     progress_bar.progress(80)
#     status_text.text("✨ Generating AI summary...")
    
#     # Generate summary with the new parameters
#     summary = generate_summary(predictions, user_type, gemini_model, systemic_alert, primary_diagnosis)
    
#     progress_bar.progress(100)
#     status_text.text("✅ Analysis complete!")
#     time.sleep(0.3)
#     progress_bar.empty()
#     status_text.empty()
    
#     # Store for chatbot
#     st.session_state.analysis_report = analysis_report
#     st.session_state.predictions = predictions
    
#     # Clean up temp file
#     try:
#         os.remove(temp_image_path)
#     except:
#         pass
    
#     # Layout: Image + Results
#     col_img, col_results = st.columns([1, 2])
    
#     with col_img:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.image(image, caption="Analyzed Image", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # Primary diagnosis card
#         st.markdown(f"""
#         <div class="card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
#             <h4>Primary Diagnosis</h4>
#             <h2>{DISEASE_FULL_NAMES.get(primary_diagnosis, primary_diagnosis)}</h2>
#             <p style="opacity: 0.9; font-size: 0.9rem;">{clinical_message}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col_results:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>📈 Detection Probabilities</h3>", unsafe_allow_html=True)
        
#         # Sort predictions by probability
#         sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
#         for disease, prob in sorted_preds:
#             color = DISEASE_COLORS.get(disease, '#95a5a6')
#             full_name = DISEASE_FULL_NAMES.get(disease, disease)
#             pct = prob * 100
            
#             # Highlight primary diagnosis and systemic diseases
#             is_systemic = disease in SYSTEMIC_DISEASES and prob > 0.3
#             is_primary = disease == primary_diagnosis
            
#             border_style = ""
#             if is_primary:
#                 border_style = "border: 3px solid #667eea;"
#             elif is_systemic:
#                 border_style = "border: 2px solid #e74c3c;"
            
#             st.markdown(f"""
#             <div class="result-bar-container" style="{border_style}">
#                 <div class="result-bar-label">
#                     <span>{'🎯 ' if is_primary else '⚠️ ' if is_systemic else ''}{full_name}</span>
#                     <span style="color: {color}; font-weight: bold;">{pct:.1f}%</span>
#                 </div>
#                 <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
#                     <div class="result-bar" style="width: {pct}%; background: {color};"></div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     # Systemic Alert - use the one from analysis report
#     if systemic_alert:
#         st.markdown(f"""
#         <div class="systemic-alert">
#             <span class="systemic-alert-icon">⚠️</span>
#             <div>
#                 <strong>Systemic Health Alert</strong><br>
#                 {clinical_message}. These findings may indicate underlying systemic conditions. 
#                 Please consult a healthcare provider for cardiovascular screening.
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # AI Summary
#     st.markdown(f"""
#     <div class="summary-box">
#         <h3>{'🤖 AI Summary for Patient' if user_type == 'Patient' else '📋 Clinical Summary'}</h3>
#         {summary}
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Action buttons
#     st.markdown("<br>", unsafe_allow_html=True)
#     col_btn1, col_btn2, col_btn3 = st.columns(3)
    
#     with col_btn1:
#         if st.button("🔄 Analyze Another Image"):
#             st.session_state.page = "upload"
#             if "chat_history" in st.session_state:
#                 del st.session_state.chat_history
#             st.rerun()
    
#     with col_btn2:
#         if st.button("🏠 Back to Home"):
#             st.session_state.page = "home"
#             if "chat_history" in st.session_state:
#                 del st.session_state.chat_history
#             st.rerun()
    
#     with col_btn3:
#         if st.button("💬 Open Chat Assistant"):
#             st.session_state.show_sidebar = True
#             st.rerun()


# def render_chatbot_sidebar():
#     """Render the chatbot sidebar."""
#     analysis_report = st.session_state.get("analysis_report", {})
#     user_type = st.session_state.get("user_type", "Patient")
#     gemini_model = setup_gemini()
    
#     with st.sidebar:
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem;">
#             <h2>💬 Ask RetiSense</h2>
#             <p style="color: #aaa;">AI-powered follow-up assistant</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Show primary diagnosis context
#         if analysis_report:
#             st.info(f"📋 Primary: {analysis_report.get('primary_diagnosis', 'N/A')}")
        
#         # Initialize chat history
#         if "chat_history" not in st.session_state:
#             st.session_state.chat_history = []
        
#         # Chat container
#         chat_container = st.container()
        
#         with chat_container:
#             for i, msg in enumerate(st.session_state.chat_history):
#                 if i % 2 == 0:
#                     st.markdown(f"""
#                     <div style="background: #667eea; color: white; padding: 0.75rem 1rem; 
#                          border-radius: 15px 15px 5px 15px; margin: 0.5rem 0; margin-left: 20%;">
#                         {msg}
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div style="background: #f0f2f6; color: #333; padding: 0.75rem 1rem; 
#                          border-radius: 15px 15px 15px 5px; margin: 0.5rem 0; margin-right: 20%;">
#                         {msg}
#                     </div>
#                     """, unsafe_allow_html=True)
        
#         # Chat input
#         st.markdown("<br>", unsafe_allow_html=True)
#         user_input = st.text_input("Ask a question...", key="chat_input", label_visibility="collapsed", 
#                                     placeholder="Ask about your results...")
        
#         if st.button("Send", use_container_width=True) and user_input:
#             st.session_state.chat_history.append(user_input)
            
#             response = get_chatbot_response(
#                 user_input, analysis_report, st.session_state.chat_history, 
#                 user_type, gemini_model
#             )
#             st.session_state.chat_history.append(response)
#             st.rerun()
        
#         # Quick questions
#         st.markdown("<br>", unsafe_allow_html=True)
#         st.markdown("**Quick Questions:**")
        
#         quick_qs = [
#             "What do these results mean?",
#             "Should I be worried?",
#             "What should I do next?",
#             "Explain the highest readings"
#         ]
        
#         for q in quick_qs:
#             if st.button(q, key=f"quick_{q[:10]}", use_container_width=True):
#                 st.session_state.chat_history.append(q)
#                 response = get_chatbot_response(
#                     q, analysis_report, st.session_state.chat_history,
#                     user_type, gemini_model
#                 )
#                 st.session_state.chat_history.append(response)
#                 st.rerun()
        
#         # Clear chat
#         st.markdown("<br>", unsafe_allow_html=True)
#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.chat_history = []
#             st.rerun()


# # ============== MAIN APP ==============
# def main():
#     load_css()
    
#     # Initialize session state
#     if "page" not in st.session_state:
#         st.session_state.page = "home"
    
#     # Render chatbot sidebar if on results page
#     if st.session_state.page == "results":
#         render_chatbot_sidebar()
    
#     # Route to appropriate page
#     if st.session_state.page == "home":
#         render_home_page()
#     elif st.session_state.page == "select_user":
#         render_user_selection()
#     elif st.session_state.page == "upload":
#         render_upload_page()
#     elif st.session_state.page == "results":
#         render_results_page()


# if __name__ == "__main__":
#     main()


"""
BlindSpot: AI-Powered Retinal Disease Screening
======================================================
A Streamlit application for multi-disease detection from fundus images.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import time
import os
import tempfile
from dotenv import load_dotenv
# import google.generativeai as genai
from google import genai

# Import the actual inference function from inference.py
from inference import run_retisense_analysis

# Load environment variables
load_dotenv()

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="BlindSpot | AI Retinal Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CONSTANTS ==============
DISEASE_LABELS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
DISEASE_FULL_NAMES = {
    'Normal': 'Normal/Healthy',
    'Diabetes': 'Diabetic Retinopathy',
    'Glaucoma': 'Glaucoma',
    'Cataract': 'Cataract',
    'AMD': 'Age-related Macular Degeneration',
    'Hypertension': 'Hypertensive Retinopathy',
    'Myopia': 'Pathological Myopia',
    'Other': 'Other Abnormalities'
}
DISEASE_COLORS = {
    'Normal': '#2ecc71',      # Green
    'Diabetes': '#e74c3c',    # Red
    'Glaucoma': '#9b59b6',    # Purple
    'Cataract': '#3498db',    # Blue
    'AMD': '#f39c12',         # Orange
    'Hypertension': '#e91e63', # Pink
    'Myopia': '#00bcd4',      # Cyan
    'Other': '#95a5a6'        # Gray
}
SYSTEMIC_DISEASES = ['Diabetes', 'Hypertension']

# Model path - pointing to your V2 3-epoch champion weights
MODEL_PATH = "checkpoints/retisense_final_v2.pth"


# ============== CUSTOM CSS ==============
def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .card-dark {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    
    /* User type selection cards */
    .user-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 3px solid transparent;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .user-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .user-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .user-card-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Result bars */
    .result-bar-container {
        margin: 1rem 0;
        background: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .result-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .result-bar {
        height: 25px;
        border-radius: 12px;
        transition: width 0.5s ease;
    }
    
    /* Alert boxes */
    .systemic-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .systemic-alert-icon {
        font-size: 2rem;
    }
    
    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        line-height: 1.8;
    }
    
    .summary-box h3 {
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chatbot styling */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 85%;
    }
    
    .chat-user {
        background: #667eea;
        color: white;
        margin-left: auto;
    }
    
    .chat-assistant {
        background: #f0f2f6;
        color: #333;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Upload area */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: #f8f9ff;
        margin: 2rem 0;
    }
    
    /* Progress styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    </style>
    """, unsafe_allow_html=True)


# ============== RUN INFERENCE ==============
def run_analysis(image_path: str) -> dict:
    """
    Run the actual BlindSpot model inference.
    FIX: Enforces absolute pathing so Streamlit never loses the weights.
    """
    abs_model_path = os.path.abspath(MODEL_PATH)
    print(f"🔥 DEBUG: Loading weights from strictly -> {abs_model_path}")
    return run_retisense_analysis(image_path, model_path=abs_model_path)


# ============== GEMINI SETUP ==============
def setup_gemini():
    """Configure Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        # THE FIX: Try the 'latest' tag, or fallback to the standard Pro model
        return genai.GenerativeModel('gemini-1.5-flash-latest') 
    return None


# ============== GEMINI SUMMARY GENERATION ==============
# ============== GEMINI SETUP ==============
def setup_gemini():
    """Configure Gemini API using the new SDK."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        # NEW SDK: We create a client object
        return genai.Client(api_key=api_key)
    return None


# # ============== GEMINI SUMMARY GENERATION ==============
# def generate_summary(predictions, user_type, client, systemic_alert=False, primary_diagnosis="Unknown"):
#     """Generate AI summary based on predictions and user type."""
#     if client is None:
#         return "⚠️ Gemini API not configured. Please add GEMINI_API_KEY to .env file."
    
#     results_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in predictions.items()])
    
#     if user_type == "Patient":
#         prompt = f"""You are a caring and empathetic medical assistant explaining eye scan results to a patient.

# The AI analysis of the retinal scan shows:
# {results_str}

# Primary Finding: {primary_diagnosis}
# {"⚠️ IMPORTANT: A systemic health alert was triggered. This may indicate underlying conditions that affect the whole body, not just the eyes." if systemic_alert else ""}

# Please:
# 1. Explain what the primary finding ({primary_diagnosis}) means in simple, everyday language
# 2. Use helpful analogies (e.g., comparing a cataract to looking through a foggy window)
# 3. If systemic indicators (Diabetes/Hypertension) are elevated, gently explain that eye blood vessels can reflect overall health
# 4. Reassure the patient while encouraging them to follow up with their doctor
# 5. Keep it warm, supportive, and easy to understand
# 6. End with a brief disclaimer that this is AI-assisted and requires professional medical confirmation

# Keep the response concise but thorough (about 150-200 words)."""
#     else:
#         prompt = f"""You are a senior ophthalmology consultant providing a clinical summary to a fellow healthcare professional.

# AI Diagnostic Screening Results:
# {results_str}

# Primary Finding: {primary_diagnosis}
# {"⚠️ SYSTEMIC ALERT: Elevated probability for systemic indicators detected." if systemic_alert else ""}

# Please provide:
# 1. A concise clinical interpretation using appropriate medical terminology
# 2. Highlight the most clinically significant findings (probabilities > 30%)
# 3. For systemic indicators, note relevant follow-up recommendations (e.g., HbA1c, BP monitoring)
# 4. Suggest differential diagnoses if applicable
# 5. Recommend appropriate referrals or additional imaging (OCT, visual fields, etc.)
# 6. Use abbreviations and terminology appropriate for medical documentation

# Format as a brief clinical note (about 150-200 words)."""

#     try:
#         # NEW SDK SYNTAX using the available gemini-2.5-flash model
#         response = client.models.generate_content(
#             model='gemini-2.5-flash',
#             contents=prompt
#         )
#         return response.text
#     except Exception as e:
#         return f"Error generating summary: {e}"

# ============== GEMINI SUMMARY GENERATION ==============
def generate_summary(predictions, user_type, client, systemic_alert=False, primary_diagnosis="Unknown"):
    """Generate AI summary based on predictions and user type."""
    if client is None:
        return "⚠️ Gemini API not configured. Please add GEMINI_API_KEY to .env file."
    
    results_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in predictions.items()])
    
    if user_type == "Patient":
        prompt = f"""You are an AI medical assistant explaining retinal scan results.

SCAN RESULTS:
{results_str}
Primary Finding: {primary_diagnosis}
Systemic Alert: {'Triggered' if systemic_alert else 'None'}

STRICT RULES:
1. NO GREETINGS OR NAMES: Do not say "Dear Patient" or use placeholders like [Patient Name]. Start the explanation immediately.
2. ZERO JARGON: Write at a 6th-grade reading level. Use simple everyday analogies. 
3. ULTRA-CONCISE: Maximum 3 short paragraphs. Do not ramble.
4. SYSTEMIC TIE-IN: If the alert is triggered, briefly explain that tiny eye blood vessels act as a window to overall heart/body health.
5. CLOSING: End strictly with: "Please share these AI screening results with your healthcare provider." """

    else:
        prompt = f"""You are an AI generating a clinical note for an ophthalmologist.

SCAN RESULTS:
{results_str}
Primary Finding: {primary_diagnosis}
Systemic Alert: {'Triggered' if systemic_alert else 'None'}

STRICT RULES:
1. NO GREETINGS OR NAMES: Do not say "Dear Doctor" or use placeholders. Begin the clinical note immediately.
2. NO FLUFF: Do not use conversational filler. Be highly concise and objective.
3. FORMAT: Use a short, bulleted list. Maximum 100 words total.
4. CONTENT: State the primary finding, note significant probabilities (>30%), and bullet the standard next steps (e.g., OCT, HbA1c, BP monitoring)."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"
    


# ============== CHATBOT RESPONSE ==============
# def get_chatbot_response(query, analysis_report, chat_history, user_type, client):
#     """Generate chatbot response for follow-up questions."""
#     if client is None:
#         return "⚠️ Chatbot requires Gemini API. Please configure GEMINI_API_KEY."
    
#     predictions = analysis_report.get('predictions', {})
#     results_context = ", ".join([f"{k}: {v*100:.1f}%" for k, v in predictions.items()])
    
#     persona = "empathetic medical assistant using simple language" if user_type == "Patient" else "clinical ophthalmology consultant using medical terminology"
    
#     history_text = "\n".join([f"{'User' if i%2==0 else 'Assistant'}: {msg}" for i, msg in enumerate(chat_history[-6:])])
    
#     prompt = f"""You are a {persona} helping with follow-up questions about retinal scan results.

# Scan Analysis:
# - Predictions: {results_context}
# - Primary Diagnosis: {analysis_report.get('primary_diagnosis', 'Unknown')}
# - Systemic Alert: {'Yes' if analysis_report.get('systemic_alert', False) else 'No'}
# - Clinical Note: {analysis_report.get('message', '')}

# Recent conversation:
# {history_text}

# User's new question: {query}

# Provide a helpful, accurate response. {"Use simple language and analogies." if user_type == "Patient" else "Use appropriate medical terminology."} 
# If asked about treatment, always recommend consulting with their healthcare provider.
# Keep responses concise (2-3 paragraphs max)."""
    
#     try:
#         # NEW SDK SYNTAX
#         response = client.models.generate_content(
#             model='gemini-2.5-flash',
#             contents=prompt
#         )
#         return response.text
#     except Exception as e:
#         return f"Error: {e}"

# ============== CHATBOT RESPONSE ==============
def get_chatbot_response(query, analysis_report, chat_history, user_type, client):
    """Generate chatbot response for follow-up questions."""
    if client is None:
        return "⚠️ Chatbot requires Gemini API."
    
    predictions = analysis_report.get('predictions', {})
    results_context = ", ".join([f"{k}: {v*100:.1f}%" for k, v in predictions.items()])
    
    if user_type == "Patient":
        persona_rules = """
        ROLE: Patient Assistant. 
        RULES: No medical jargon. No rambling. Limit to 2 short paragraphs.
        DO NOT use greetings, sign-offs, or placeholders like [Name]. Answer the question directly and simply.
        """
    else:
        persona_rules = """
        ROLE: Clinical AI Assistant for Doctors or Medical Professionals.
        RULES: Concise, objective medical terminology. No conversational fluff or greetings. Bullet points preferred. Maximum 75 words.
        """
    
    history_text = "\n".join([f"{'User' if i%2==0 else 'Assistant'}: {msg}" for i, msg in enumerate(chat_history[-6:])])
    
    prompt = f"""{persona_rules}

CONTEXT:
Predictions: {results_context}
Primary: {analysis_report.get('primary_diagnosis', 'Unknown')}
Systemic Alert: {'Yes' if analysis_report.get('systemic_alert', False) else 'No'}

CHAT HISTORY:
{history_text}

USER QUESTION: {query}
RESPONSE:"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# # ============== UI COMPONENTS ==============
# def render_home_page():
#     """Render the landing/home page."""
#     st.markdown("""
#     <div class="main-header">
#         <h1>👁️ BlindSpot</h1>
#         <p>AI-Powered Multi-Disease Retinal Screening System</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col2:
#         st.markdown("""
#         <div class="card" style="text-align: center;">
#             <h2>🔬 Advanced AI Screening</h2>
#             <p style="font-size: 1.1rem; color: #666; margin: 1.5rem 0;">
#                 Our state-of-the-art AI model, powered by <strong>RETFound</strong> 
#                 (trained on 1.6 million retinal images), can detect <strong>8 different conditions</strong> 
#                 from a single fundus photograph.
#             </p>
            
#             <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin: 2rem 0;">
#                 <span style="background: #e8f5e9; color: #2e7d32; padding: 0.5rem 1rem; border-radius: 20px;">✓ Diabetic Retinopathy</span>
#                 <span style="background: #f3e5f5; color: #7b1fa2; padding: 0.5rem 1rem; border-radius: 20px;">✓ Glaucoma</span>
#                 <span style="background: #e3f2fd; color: #1565c0; padding: 0.5rem 1rem; border-radius: 20px;">✓ Cataract</span>
#                 <span style="background: #fff3e0; color: #ef6c00; padding: 0.5rem 1rem; border-radius: 20px;">✓ AMD</span>
#                 <span style="background: #fce4ec; color: #c2185b; padding: 0.5rem 1rem; border-radius: 20px;">✓ Hypertension</span>
#                 <span style="background: #e0f7fa; color: #00838f; padding: 0.5rem 1rem; border-radius: 20px;">✓ Myopia</span>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown("<br>", unsafe_allow_html=True)
        
#         col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
#         with col_btn2:
#             if st.button("🚀 Try It Now", use_container_width=True, key="try_it"):
#                 st.session_state.page = "select_user"
#                 st.rerun()
        
#         st.markdown("""
#         <div style="text-align: center; margin-top: 2rem; color: #999; font-size: 0.9rem;">
#             <p>⚠️ This tool is for screening purposes only and does not replace professional medical diagnosis.</p>
#         </div>
#         """, unsafe_allow_html=True)

def render_home_page():
    """Render the landing/home page."""
    # Header Section
    st.markdown("""
<div class="main-header">
    <h1>BlindSpot</h1>
    <p>AI-Powered Multi-Disease Retinal Screening System</p>
</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 1. Text Description
        st.markdown("""
<div style="text-align: center; background: white; border-radius: 15px; padding: 2rem; border: 1px solid #e0e0e0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <h2 style="color: #333;">🔬 Advanced AI Screening</h2>
    <p style="font-size: 1.1rem; color: #666; margin: 1.5rem 0;">
        Our state-of-the-art AI model, powered by <strong>RETFound</strong> 
        (trained on 1.6 million retinal images), can detect <strong>8 different conditions</strong> 
        from a single fundus photograph.
    </p>
</div>
""", unsafe_allow_html=True)

        # 2. Colorful Badges (THE FIX: NO LEADING INDENTATION INSIDE QUOTES)
        badges_html = """
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin: 2rem 0;">
    <span style="background: #e8f5e9; color: #2e7d32; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; font-family: sans-serif;">✓ Diabetic Retinopathy</span>
    <span style="background: #f3e5f5; color: #7b1fa2; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; font-family: sans-serif;">✓ Glaucoma</span>
    <span style="background: #e3f2fd; color: #1565c0; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; font-family: sans-serif;">✓ Cataract</span>
    <span style="background: #fff3e0; color: #ef6c00; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; font-family: sans-serif;">✓ AMD</span>
    <span style="background: #fce4ec; color: #c2185b; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; font-family: sans-serif;">✓ Hypertension</span>
    <span style="background: #e0f7fa; color: #00838f; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; font-family: sans-serif;">✓ Myopia</span>
</div>
"""
        st.markdown(badges_html, unsafe_allow_html=True)
        
        # 3. Action Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 Try It Now", use_container_width=True, key="try_it"):
                st.session_state.page = "select_user"
                st.rerun()
        
        # 4. Disclaimer
        st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #999; font-size: 0.9rem;">
    <p>⚠️ This tool is for screening purposes only and does not replace professional medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)

def render_user_selection():
    """Render user type selection page."""
    st.markdown("""
    <div class="main-header">
        <h1>👤 Who's Using BlindSpot?</h1>
        <p>Select your profile for a personalized experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center; height: 300px;">
            <div style="font-size: 5rem;">🧑‍🤝‍🧑</div>
            <h3>Patient</h3>
            <p style="color: #666;">Get easy-to-understand results explained in plain English</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("I'm a Patient", use_container_width=True, key="patient_btn"):
            st.session_state.user_type = "Patient"
            st.session_state.page = "upload"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="card" style="text-align: center; height: 300px;">
            <div style="font-size: 5rem;">👨‍⚕️</div>
            <h3>Medical Professional</h3>
            <p style="color: #666;">Get detailed clinical analysis with medical terminology</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("I'm a Doctor/Professional", use_container_width=True, key="doctor_btn"):
            st.session_state.user_type = "Doctor"
            st.session_state.page = "upload"
            st.rerun()
    
    with col2:
        st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)
    
    # Back button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

def render_upload_page():
    """Render image upload page."""
    user_type = st.session_state.get("user_type", "Patient")
    
    st.markdown(f"""
    <div class="main-header">
        <h1>📤 Upload Retinal Image</h1>
        <p>Upload a fundus photograph for AI analysis • Mode: {user_type}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center;">📷 Select Image</h3>
            <p style="text-align: center; color: #666;">
                Upload a high-quality fundus photograph (JPG, PNG, JPEG)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a fundus image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("🔍 Analyze Image", use_container_width=True):
                # 1. Save Image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name, format="PNG")
                    temp_image_path = tmp_file.name
                
                # ==========================================================
                # THE FIX: DO ALL THE HEAVY AI WORK HERE BEFORE SWITCHING PAGES
                # ==========================================================
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🧠 Running RETFound AI analysis...")
                progress_bar.progress(30)
                
                # Run Model
                try:
                    analysis_report = run_analysis(temp_image_path)
                except Exception as e:
                    st.error(f"Error running model: {e}")
                    st.stop()
                
                status_text.text("✨ Generating Gemini Summary...")
                progress_bar.progress(70)
                
                # Generate Summary
                client = setup_gemini()
                summary = generate_summary(
                    analysis_report['predictions'], 
                    user_type, 
                    client, 
                    analysis_report['systemic_alert'], 
                    analysis_report['primary_diagnosis']
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                time.sleep(0.5)
                
                # 2. Store EVERYTHING safely in memory
                st.session_state.uploaded_image = image
                st.session_state.analysis_report = analysis_report
                st.session_state.summary = summary
                
                # Clear chat history for the new image
                if "chat_history" in st.session_state:
                    st.session_state.chat_history = []
                
                # 3. Switch to Results Page
                st.session_state.page = "results"
                st.rerun()
    
    # Back button
    st.markdown("<br>", unsafe_allow_html=True)
    col_back, _, _ = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Change User Type"):
            st.session_state.page = "select_user"
            st.rerun()
# def render_upload_page():
#     """Render image upload page."""
#     user_type = st.session_state.get("user_type", "Patient")
    
#     st.markdown(f"""
#     <div class="main-header">
#         <h1>📤 Upload Retinal Image</h1>
#         <p>Upload a fundus photograph for AI analysis • Mode: {user_type}</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col2:
#         st.markdown("""
#         <div class="card">
#             <h3 style="text-align: center;">📷 Select Image</h3>
#             <p style="text-align: center; color: #666;">
#                 Upload a high-quality fundus photograph (JPG, PNG, JPEG)
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         uploaded_file = st.file_uploader(
#             "Choose a fundus image",
#             type=["jpg", "jpeg", "png"],
#             label_visibility="collapsed"
#         )
        
#         if uploaded_file:
#             # FIX: Force RGB conversion to strip alpha channels and web metadata
#             image = Image.open(uploaded_file).convert('RGB')
            
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.image(image, caption="Uploaded Image", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
            
#             if st.button("🔍 Analyze Image", use_container_width=True):
#                 # FIX: Save losslessly as PNG so pixel values don't shift during compression
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#                     image.save(tmp_file.name, format="PNG")
#                     st.session_state.temp_image_path = tmp_file.name
#                 st.session_state.uploaded_image = image
#                 st.session_state.page = "results"
#                 st.rerun()
    
#     # Back button
#     st.markdown("<br>", unsafe_allow_html=True)
#     col_back, _, _ = st.columns([1, 2, 1])
#     with col_back:
#         if st.button("← Change User Type"):
#             st.session_state.page = "select_user"
#             st.rerun()

def render_results_page():
    """Render results page with static analysis."""
    # Pull everything directly from memory. No AI generation happens here!
    user_type = st.session_state.get("user_type", "Patient")
    image = st.session_state.get("uploaded_image")
    analysis_report = st.session_state.get("analysis_report")
    summary = st.session_state.get("summary")
    
    if image is None or analysis_report is None:
        st.session_state.page = "upload"
        st.rerun()
        return
        
    predictions = analysis_report['predictions']
    primary_diagnosis = analysis_report['primary_diagnosis']
    systemic_alert = analysis_report['systemic_alert']
    clinical_message = analysis_report['message']
    
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Analysis Results</h1>
        <p>AI-Powered Screening Report • {user_type} View</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout: Image + Results
    col_img, col_results = st.columns([1, 2])
    
    with col_img:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="Analyzed Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h4>Primary Diagnosis</h4>
            <h2>{DISEASE_FULL_NAMES.get(primary_diagnosis, primary_diagnosis)}</h2>
            <p style="opacity: 0.9; font-size: 0.9rem;">{clinical_message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_results:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>📈 Detection Probabilities</h3>", unsafe_allow_html=True)
        
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for disease, prob in sorted_preds:
            color = DISEASE_COLORS.get(disease, '#95a5a6')
            full_name = DISEASE_FULL_NAMES.get(disease, disease)
            pct = prob * 100
            
            is_systemic = disease in SYSTEMIC_DISEASES and prob > 0.3
            is_primary = disease == primary_diagnosis
            
            border_style = "border: 3px solid #667eea;" if is_primary else "border: 2px solid #e74c3c;" if is_systemic else ""
            
            st.markdown(f"""
            <div class="result-bar-container" style="{border_style}">
                <div class="result-bar-label">
                    <span>{'🎯 ' if is_primary else '⚠️ ' if is_systemic else ''}{full_name}</span>
                    <span style="color: {color}; font-weight: bold;">{pct:.1f}%</span>
                </div>
                <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                    <div class="result-bar" style="width: {pct}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if systemic_alert:
        st.markdown(f"""
        <div class="systemic-alert">
            <span class="systemic-alert-icon">⚠️</span>
            <div>
                <strong>Systemic Health Alert</strong><br>
                {clinical_message}. These findings may indicate underlying systemic conditions. 
                Please consult a healthcare provider for cardiovascular screening.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # This text is safely locked in memory, so it will instantly appear without regenerating!
    st.markdown(f"""
    <div class="summary-box">
        <h3>{'🤖 AI Summary for Patient' if user_type == 'Patient' else '📋 Clinical Summary'}</h3>
        {summary}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔄 Analyze Another Image", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
    
    with col_btn2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
# def render_results_page():
#     """Render results page with analysis."""
#     user_type = st.session_state.get("user_type", "Patient")
#     image = st.session_state.get("uploaded_image")
#     temp_image_path = st.session_state.get("temp_image_path")
    
#     if image is None or temp_image_path is None:
#         st.session_state.page = "upload"
#         st.rerun()
#         return
    
#     st.markdown(f"""
#     <div class="main-header">
#         <h1>📊 Analysis Results</h1>
#         <p>AI-Powered Screening Report • {user_type} View</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     client = setup_gemini()
    
#     # ================= THE ULTIMATE STATIC LOCK =================
#     # Only run the models if we haven't saved the report and summary to memory yet!
#     if "analysis_report" not in st.session_state or "summary" not in st.session_state:
        
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         status_text.text("🔄 Loading RetiSense model...")
#         progress_bar.progress(20)
        
#         status_text.text("🧠 Running AI analysis (using trained model)...")
#         progress_bar.progress(50)
        
#         # Run the ACTUAL inference
#         try:
#             analysis_report = run_analysis(temp_image_path)
#         except Exception as e:
#             st.error(f"Error running model: {e}")
#             st.stop()
        
#         progress_bar.progress(80)
#         status_text.text("✨ Generating AI summary...")
        
#         # Generate summary 
#         predictions = analysis_report['predictions']
#         primary_diagnosis = analysis_report['primary_diagnosis']
#         systemic_alert = analysis_report['systemic_alert']
        
#         summary = generate_summary(predictions, user_type, client, systemic_alert, primary_diagnosis)
        
#         progress_bar.progress(100)
#         status_text.text("✅ Analysis complete!")
#         time.sleep(0.3)
#         progress_bar.empty()
#         status_text.empty()
        
#         # SAVE DIRECTLY TO MEMORY
#         st.session_state.analysis_report = analysis_report
#         st.session_state.summary = summary
#     # ============================================================

#     # Pull the static, already-generated data from memory!
#     analysis_report = st.session_state.analysis_report
#     predictions = analysis_report['predictions']
#     summary = st.session_state.summary
#     primary_diagnosis = analysis_report['primary_diagnosis']
#     systemic_alert = analysis_report['systemic_alert']
#     clinical_message = analysis_report['message']
    
#     # Layout: Image + Results
#     col_img, col_results = st.columns([1, 2])
    
#     with col_img:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.image(image, caption="Analyzed Image", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # Primary diagnosis card
#         st.markdown(f"""
#         <div class="card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
#             <h4>Primary Diagnosis</h4>
#             <h2>{DISEASE_FULL_NAMES.get(primary_diagnosis, primary_diagnosis)}</h2>
#             <p style="opacity: 0.9; font-size: 0.9rem;">{clinical_message}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col_results:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>📈 Detection Probabilities</h3>", unsafe_allow_html=True)
        
#         # Sort predictions by probability
#         sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
#         for disease, prob in sorted_preds:
#             color = DISEASE_COLORS.get(disease, '#95a5a6')
#             full_name = DISEASE_FULL_NAMES.get(disease, disease)
#             pct = prob * 100
            
#             is_systemic = disease in SYSTEMIC_DISEASES and prob > 0.3
#             is_primary = disease == primary_diagnosis
            
#             border_style = ""
#             if is_primary:
#                 border_style = "border: 3px solid #667eea;"
#             elif is_systemic:
#                 border_style = "border: 2px solid #e74c3c;"
            
#             st.markdown(f"""
#             <div class="result-bar-container" style="{border_style}">
#                 <div class="result-bar-label">
#                     <span>{'🎯 ' if is_primary else '⚠️ ' if is_systemic else ''}{full_name}</span>
#                     <span style="color: {color}; font-weight: bold;">{pct:.1f}%</span>
#                 </div>
#                 <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
#                     <div class="result-bar" style="width: {pct}%; background: {color};"></div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     # Systemic Alert
#     if systemic_alert:
#         st.markdown(f"""
#         <div class="systemic-alert">
#             <span class="systemic-alert-icon">⚠️</span>
#             <div>
#                 <strong>Systemic Health Alert</strong><br>
#                 {clinical_message}. These findings may indicate underlying systemic conditions. 
#                 Please consult a healthcare provider for cardiovascular screening.
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # AI Summary
#     st.markdown(f"""
#     <div class="summary-box">
#         <h3>{'🤖 AI Summary for Patient' if user_type == 'Patient' else '📋 Clinical Summary'}</h3>
#         {summary}
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Action buttons
#     st.markdown("<br>", unsafe_allow_html=True)
#     col_btn1, col_btn2, col_btn3 = st.columns(3)
    
#     with col_btn1:
#         if st.button("🔄 Analyze Another Image"):
#             st.session_state.page = "upload"
#             # Delete the locks so the NEXT image can run
#             for key in ["analysis_report", "summary", "chat_history"]:
#                 if key in st.session_state:
#                     del st.session_state[key]
#             st.rerun()
    
#     with col_btn2:
#         if st.button("🏠 Back to Home"):
#             st.session_state.page = "home"
#             # Delete the locks so the NEXT image can run
#             for key in ["analysis_report", "summary", "chat_history"]:
#                 if key in st.session_state:
#                     del st.session_state[key]
#             st.rerun()
    
#     with col_btn3:
#         if st.button("💬 Open Chat Assistant"):
#             st.session_state.show_sidebar = True
#             st.rerun()


# def render_results_page():
#     """Render results page with analysis."""
#     user_type = st.session_state.get("user_type", "Patient")
#     image = st.session_state.get("uploaded_image")
#     temp_image_path = st.session_state.get("temp_image_path")
    
#     if image is None or temp_image_path is None:
#         st.session_state.page = "upload"
#         st.rerun()
#         return
    
#     st.markdown(f"""
#     <div class="main-header">
#         <h1>📊 Analysis Results</h1>
#         <p>AI-Powered Screening Report • {user_type} View</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     client = setup_gemini()
    
#     # ================= THE FIX =================
#     # Only run the heavy model and API calls if we haven't done it yet!
#     if "analysis_report" not in st.session_state:
#         # Show progress bar during analysis
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         status_text.text("🔄 Loading RetiSense model...")
#         progress_bar.progress(20)
        
#         status_text.text("🧠 Running AI analysis (using trained model)...")
#         progress_bar.progress(50)
        
#         # Run the ACTUAL inference using inference.py
#         try:
#             analysis_report = run_analysis(temp_image_path)
#         except Exception as e:
#             st.error(f"Error running model: {e}")
#             st.stop()
        
#         progress_bar.progress(80)
#         status_text.text("✨ Generating AI summary...")
        
#         # Generate summary 
#         predictions = analysis_report['predictions']
#         primary_diagnosis = analysis_report['primary_diagnosis']
#         systemic_alert = analysis_report['systemic_alert']
        
#         summary = generate_summary(predictions, user_type, client, systemic_alert, primary_diagnosis)
        
#         progress_bar.progress(100)
#         status_text.text("✅ Analysis complete!")
#         time.sleep(0.3)
#         progress_bar.empty()
#         status_text.empty()
        
#         # SAVE TO MEMORY so it survives chat reruns
#         st.session_state.analysis_report = analysis_report
#         st.session_state.predictions = predictions
#         st.session_state.summary = summary
        
#         # Clean up temp file safely ONLY after the first run
#         try:
#             if os.path.exists(temp_image_path):
#                 os.remove(temp_image_path)
#         except:
#             pass

#     # Retrieve from memory on reruns (like when chatting)
#     analysis_report = st.session_state.analysis_report
#     predictions = st.session_state.predictions
#     summary = st.session_state.summary
#     primary_diagnosis = analysis_report['primary_diagnosis']
#     systemic_alert = analysis_report['systemic_alert']
#     clinical_message = analysis_report['message']
#     # ===========================================
    
#     # Layout: Image + Results
#     col_img, col_results = st.columns([1, 2])
    
#     with col_img:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.image(image, caption="Analyzed Image", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # Primary diagnosis card
#         st.markdown(f"""
#         <div class="card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
#             <h4>Primary Diagnosis</h4>
#             <h2>{DISEASE_FULL_NAMES.get(primary_diagnosis, primary_diagnosis)}</h2>
#             <p style="opacity: 0.9; font-size: 0.9rem;">{clinical_message}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col_results:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>📈 Detection Probabilities</h3>", unsafe_allow_html=True)
        
#         # Sort predictions by probability
#         sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
#         for disease, prob in sorted_preds:
#             color = DISEASE_COLORS.get(disease, '#95a5a6')
#             full_name = DISEASE_FULL_NAMES.get(disease, disease)
#             pct = prob * 100
            
#             # Highlight primary diagnosis and systemic diseases
#             is_systemic = disease in SYSTEMIC_DISEASES and prob > 0.3
#             is_primary = disease == primary_diagnosis
            
#             border_style = ""
#             if is_primary:
#                 border_style = "border: 3px solid #667eea;"
#             elif is_systemic:
#                 border_style = "border: 2px solid #e74c3c;"
            
#             st.markdown(f"""
#             <div class="result-bar-container" style="{border_style}">
#                 <div class="result-bar-label">
#                     <span>{'🎯 ' if is_primary else '⚠️ ' if is_systemic else ''}{full_name}</span>
#                     <span style="color: {color}; font-weight: bold;">{pct:.1f}%</span>
#                 </div>
#                 <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
#                     <div class="result-bar" style="width: {pct}%; background: {color};"></div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     # Systemic Alert
#     if systemic_alert:
#         st.markdown(f"""
#         <div class="systemic-alert">
#             <span class="systemic-alert-icon">⚠️</span>
#             <div>
#                 <strong>Systemic Health Alert</strong><br>
#                 {clinical_message}. These findings may indicate underlying systemic conditions. 
#                 Please consult a healthcare provider for cardiovascular screening.
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # AI Summary (Now pulled from memory instantly!)
#     st.markdown(f"""
#     <div class="summary-box">
#         <h3>{'🤖 AI Summary for Patient' if user_type == 'Patient' else '📋 Clinical Summary'}</h3>
#         {summary}
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Action buttons
#     st.markdown("<br>", unsafe_allow_html=True)
#     col_btn1, col_btn2, col_btn3 = st.columns(3)
    
#     with col_btn1:
#         if st.button("🔄 Analyze Another Image"):
#             st.session_state.page = "upload"
#             # CLEAR MEMORY SO THE NEXT IMAGE WORKS
#             for key in ["chat_history", "analysis_report", "predictions", "summary"]:
#                 if key in st.session_state:
#                     del st.session_state[key]
#             st.rerun()
    
#     with col_btn2:
#         if st.button("🏠 Back to Home"):
#             st.session_state.page = "home"
#             # CLEAR MEMORY
#             for key in ["chat_history", "analysis_report", "predictions", "summary"]:
#                 if key in st.session_state:
#                     del st.session_state[key]
#             st.rerun()
    
#     with col_btn3:
#         if st.button("💬 Open Chat Assistant"):
#             st.session_state.show_sidebar = True
#             st.rerun()





# def render_results_page():
#     """Render results page with analysis."""
#     user_type = st.session_state.get("user_type", "Patient")
#     image = st.session_state.get("uploaded_image")
#     temp_image_path = st.session_state.get("temp_image_path")
    
#     if image is None or temp_image_path is None:
#         st.session_state.page = "upload"
#         st.rerun()
#         return
    
#     st.markdown(f"""
#     <div class="main-header">
#         <h1>📊 Analysis Results</h1>
#         <p>AI-Powered Screening Report • {user_type} View</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     gemini_model = setup_gemini()
    
#     # Show progress bar during analysis
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     status_text.text("🔄 Loading RetiSense model...")
#     progress_bar.progress(20)
    
#     status_text.text("🧠 Running AI analysis (using trained model)...")
#     progress_bar.progress(50)
    
#     # Run the ACTUAL inference using inference.py
#     try:
#         analysis_report = run_analysis(temp_image_path)
#         predictions = analysis_report['predictions']
#         primary_diagnosis = analysis_report['primary_diagnosis']
#         systemic_alert = analysis_report['systemic_alert']
#         clinical_message = analysis_report['message']
#     except Exception as e:
#         st.error(f"Error running model: {e}")
#         st.stop()
    
#     progress_bar.progress(80)
#     status_text.text("✨ Generating AI summary...")
    
#     # Generate summary with the new parameters
#     summary = generate_summary(predictions, user_type, gemini_model, systemic_alert, primary_diagnosis)
    
#     progress_bar.progress(100)
#     status_text.text("✅ Analysis complete!")
#     time.sleep(0.3)
#     progress_bar.empty()
#     status_text.empty()
    
#     # Store for chatbot
#     st.session_state.analysis_report = analysis_report
#     st.session_state.predictions = predictions
    
#     # Clean up temp file
#     try:
#         os.remove(temp_image_path)
#     except:
#         pass
    
#     # Layout: Image + Results
#     col_img, col_results = st.columns([1, 2])
    
#     with col_img:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.image(image, caption="Analyzed Image", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # Primary diagnosis card
#         st.markdown(f"""
#         <div class="card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
#             <h4>Primary Diagnosis</h4>
#             <h2>{DISEASE_FULL_NAMES.get(primary_diagnosis, primary_diagnosis)}</h2>
#             <p style="opacity: 0.9; font-size: 0.9rem;">{clinical_message}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col_results:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("<h3>📈 Detection Probabilities</h3>", unsafe_allow_html=True)
        
#         # Sort predictions by probability
#         sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
#         for disease, prob in sorted_preds:
#             color = DISEASE_COLORS.get(disease, '#95a5a6')
#             full_name = DISEASE_FULL_NAMES.get(disease, disease)
#             pct = prob * 100
            
#             # Highlight primary diagnosis and systemic diseases
#             is_systemic = disease in SYSTEMIC_DISEASES and prob > 0.3
#             is_primary = disease == primary_diagnosis
            
#             border_style = ""
#             if is_primary:
#                 border_style = "border: 3px solid #667eea;"
#             elif is_systemic:
#                 border_style = "border: 2px solid #e74c3c;"
            
#             st.markdown(f"""
#             <div class="result-bar-container" style="{border_style}">
#                 <div class="result-bar-label">
#                     <span>{'🎯 ' if is_primary else '⚠️ ' if is_systemic else ''}{full_name}</span>
#                     <span style="color: {color}; font-weight: bold;">{pct:.2f}%</span>
#                 </div>
#                 <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
#                     <div class="result-bar" style="width: {pct}%; background: {color};"></div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     # Systemic Alert - use the one from analysis report
#     if systemic_alert:
#         st.markdown(f"""
#         <div class="systemic-alert">
#             <span class="systemic-alert-icon">⚠️</span>
#             <div>
#                 <strong>Systemic Health Alert</strong><br>
#                 {clinical_message}. These findings may indicate underlying systemic conditions. 
#                 Please consult a healthcare provider for cardiovascular screening.
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # AI Summary
#     st.markdown(f"""
#     <div class="summary-box">
#         <h3>{'🤖 AI Summary for Patient' if user_type == 'Patient' else '📋 Clinical Summary'}</h3>
#         {summary}
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Action buttons
#     st.markdown("<br>", unsafe_allow_html=True)
#     col_btn1, col_btn2, col_btn3 = st.columns(3)
    
#     with col_btn1:
#         if st.button("🔄 Analyze Another Image"):
#             st.session_state.page = "upload"
#             if "chat_history" in st.session_state:
#                 del st.session_state.chat_history
#             st.rerun()
    
#     with col_btn2:
#         if st.button("🏠 Back to Home"):
#             st.session_state.page = "home"
#             if "chat_history" in st.session_state:
#                 del st.session_state.chat_history
#             st.rerun()
    
#     with col_btn3:
#         if st.button("💬 Open Chat Assistant"):
#             st.session_state.show_sidebar = True
#             st.rerun()


def render_chatbot_sidebar():
    """Render the chatbot sidebar."""
    analysis_report = st.session_state.get("analysis_report", {})
    user_type = st.session_state.get("user_type", "Patient")
    gemini_model = setup_gemini()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>💬 Ask BlindSpot</h2>
            <p style="color: #aaa;">AI-powered follow-up assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show primary diagnosis context
        if analysis_report:
            st.info(f"📋 Primary: {analysis_report.get('primary_diagnosis', 'N/A')}")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for i, msg in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.markdown(f"""
                    <div style="background: #667eea; color: white; padding: 0.75rem 1rem; 
                         border-radius: 15px 15px 5px 15px; margin: 0.5rem 0; margin-left: 20%;">
                        {msg}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f0f2f6; color: #333; padding: 0.75rem 1rem; 
                         border-radius: 15px 15px 15px 5px; margin: 0.5rem 0; margin-right: 20%;">
                        {msg}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input("Ask a question...", key="chat_input", label_visibility="collapsed", 
                                    placeholder="Ask about your results...")
        
        if st.button("Send", use_container_width=True) and user_input:
            st.session_state.chat_history.append(user_input)
            
            response = get_chatbot_response(
                user_input, analysis_report, st.session_state.chat_history, 
                user_type, gemini_model
            )
            st.session_state.chat_history.append(response)
            st.rerun()
        
        # Quick questions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Quick Questions:**")
        
        quick_qs = [
            "What do these results mean?",
            "Should I be worried?",
            "What should I do next?",
            "Explain the highest readings"
        ]
        
        for q in quick_qs:
            if st.button(q, key=f"quick_{q[:10]}", use_container_width=True):
                st.session_state.chat_history.append(q)
                response = get_chatbot_response(
                    q, analysis_report, st.session_state.chat_history,
                    user_type, gemini_model
                )
                st.session_state.chat_history.append(response)
                st.rerun()
        
        # Clear chat
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ============== MAIN APP ==============
def main():
    load_css()
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Render chatbot sidebar if on results page
    if st.session_state.page == "results":
        render_chatbot_sidebar()
    
    # Route to appropriate page
    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "select_user":
        render_user_selection()
    elif st.session_state.page == "upload":
        render_upload_page()
    elif st.session_state.page == "results":
        render_results_page()


if __name__ == "__main__":
    main()