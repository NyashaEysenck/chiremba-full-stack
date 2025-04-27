 # Chiremba AI Health Platform

## Overview
Chiremba is an AI-powered healthcare assistant designed to provide users with accessible, accurate, and secure medical diagnostics and consultations. The system combines advanced machine learning models, natural language processing, and user-friendly interfaces to deliver features such as symptom checking, image-based diagnosis, and virtual consultations.

---

## Features

- **Symptom Checker:**
  - Users can input symptoms (with optional vital signs) to receive AI-generated analyses of possible conditions, urgency levels, and recommended next steps.
  - Uses Google Gemini or OpenAI models for language understanding and medical reasoning.
  - Limitations: Not a substitute for professional medical advice. Model responses may miss rare or complex conditions.

- **Image Diagnosis:**
  - Upload or capture medical images (e.g., X-rays, skin lesions) for instant AI analysis.
  - Supported models:
    - **Brain Tumor Detection** (MRI): Glioma, Meningioma, Pituitary, No Tumor
    - **Pneumonia Detection** (Chest X-ray): Normal, Pneumonia
    - **Skin Infection Classification**: Cellulitis, Athlete's Foot, Impetigo, Chickenpox, Cutaneous Larva Migrans, Nail Fungus, Ringworm, Shingles
    - **Lung Cancer Prediction** (CT): Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal
  - Each condition includes a description, urgency, and clinical recommendations.
  - Limitations: Models rely on image quality and may not generalize to atypical cases. Always consult a healthcare provider for confirmation.

- **Virtual Consultation (Chatbot):**
  - Conversational AI assistant for health questions, triage, and guidance.
  - Supports English and Shona languages.
  - Integrates text-to-speech (OpenAI, ElevenLabs) for accessibility.
  - Limitations: AI may not understand highly specialized or ambiguous queries. Responses are for informational purposes only.

- **Admin Panel:**
  - Secure management of users, roles, and permissions.
  - Staff and admin accounts for healthcare professionals.

- **Security & Privacy:**
  - Authentication and role-based access control.
  - Data is handled securely and not shared with unauthorized parties.

---

## Models Used & Limitations

### 1. **Image Analysis Models**
- **Brain Tumor:**
  - Deep learning classifier trained on MRI scans.
  - Classes: Glioma, Meningioma, Pituitary, No Tumor
  - Limitation: May not detect rare tumors or artifacts; accuracy depends on image quality.

- **Pneumonia:**
  - CNN model for chest X-ray analysis.
  - Classes: Normal, Pneumonia
  - Limitation: May miss subtle findings or non-pneumonia pathologies.

- **Skin Infection:**
  - Multi-class classifier for common skin conditions.
  - Classes: Cellulitis, Athlete's Foot, Impetigo, Chickenpox, Cutaneous Larva Migrans, Nail Fungus, Ringworm, Shingles
  - Limitation: May not recognize rare or mixed infections.

- **Lung Cancer:**
  - Model for CT-based lung cancer subtype prediction.
  - Classes: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal
  - Limitation: Not a replacement for biopsy or radiologist review.

### 2. **Language Models**
- **Gemini (Google Generative AI):**
  - Used for symptom analysis, medical reasoning, and report generation.
  - Limitation: May hallucinate or provide incomplete information. Not a medical expert.

- **OpenAI (GPT):**
  - Used for chatbot, text-to-speech, and fallback for symptom analysis.
  - Limitation: Not always up-to-date with latest medical guidelines.

---

## Environment Variables & Setup
- Requires API keys for OpenAI, Google Gemini, MongoDB, and TTS providers (see `.env` file).
- Ensure backend services are running for full functionality.

---

## Disclaimer
Chiremba is an assistive tool and **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider with any questions regarding medical conditions or symptoms.

---

## Getting Started
1. Clone the repository.
2. Install dependencies (`npm install` in both root and frontend folders).
3. Configure environment variables in `.env` files.
4. Start backend and frontend servers.
5. Access the app via the provided local URL.

---

## Contact & Contributions
For questions, support, or contributions, please contact the project maintainers or submit an issue via GitHub.