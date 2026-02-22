# 👁️ BlindSpot: Systemic Health Intelligence
**Hacklytics 2026 | Developed by Nihal**

> "The eye is the only place in the body where we can non-invasively view the microvasculature. BlindSpot turns a routine eye exam into a systemic health triage tool."

---

## 🌟 The Vision
**BlindSpot** is a medical AI portal designed to bridge the gap between ocular diagnostics and systemic health monitoring. By leveraging the **RETFound** foundation model, BlindSpot detects 8 distinct ocular conditions while simultaneously identifying markers for **Hypertension** and **Diabetes** through retinal vascular patterns.

---

## 🚀 Performance Breakthrough
Rather than training from scratch, BlindSpot utilizes **Transfer Learning** on a **Vision Transformer (ViT)** backbone pre-trained on 1.6 million retinal images. Through a specialized two-stage fine-tuning process, we achieved significant performance gains:

### **Final Metrics (Macro AUC: 0.8261)**
| Condition | AUC Score | Accuracy |
| :--- | :--- | :--- |
| **Cataract** | 0.9497 | 96.95% |
| **Myopia** | 0.9500 | 98.51% |
| **Glaucoma** | 0.8434 | 93.98% |
| **Diabetes** | 0.7490 | 71.38% |
| **Hypertension** | 0.7282 | 96.72% |



---

## 🛠️ Key Features
* **Dual-Persona Interface:** Tailored summaries for **Patients** (empathetic, layman English) and **Medical Professionals** (technical, clinical terminology).
* **Systemic Alert System:** Automatically flags cardiovascular risks detected in the retinal microvasculature.
* **Gemini-Powered Clinical Assistant:** A real-time, context-aware chatbot using **Gemini 1.5 Flash** to answer follow-up questions about diagnostic results.
* **Rapid Inference:** Optimized Stage 2 fine-tuning (last-block unfreezing) allows for analysis in under 5 seconds on consumer hardware.

---

## 🏗️ Tech Stack
* **Model Architecture:** Vision Transformer (ViT) - [RETFoundMAE](https://github.com/YukunZhou/RETFound)
* **Deep Learning:** PyTorch, Torchvision
* **Frontend:** Streamlit
* **LLM Integration:** Google Gemini 1.5 Flash API
* **Data Source:** ODIR-5K (Ocular Disease Intelligent Recognition)



---

## ⚙️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/nihalika-kaur/BlindSpot-Hacklytics-2026.git](https://github.com/nihalika-kaur/BlindSpot-Hacklytics-2026.git)
    cd BlindSpot-Hacklytics-2026
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```text
    GEMINI_API_KEY=your_google_ai_studio_key
    HF_TOKEN=your_huggingface_token
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Portal:**
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Technical Deep-Dive: Selective Unfreezing
To reach an **0.826 AUC**, we implemented a "Last-Block Fine-Tuning" strategy. By freezing the majority of the 12-layer Transformer encoder, we preserved the robust medical feature extraction of the foundation model while specializing the final attention blocks to the specific noise and lighting conditions of the ODIR-5K dataset. This prevented overfitting and ensured high generalization across diverse patient demographics.



---

## 🛡️ License & Disclaimer
This project is developed for **Hacklytics 2026**. It is a research prototype and should not be used for self-diagnosis. Always consult with a qualified healthcare professional.
