"""
Streamlit Monolithic Interface for Lung Cancer Prediction, EDA, & Clinical AI Assistant.
"""
import os
import pickle
import pandas as pd
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from langchain_groq import ChatGroq

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Prakhar | Oncology AI", page_icon="🫁")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(220, 53, 69, 0.05);
        border: 1px solid rgba(220, 53, 69, 0.2);
        padding: 5%;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .disclaimer {
        font-size: 13px;
        color: #856404;
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 5px;
        border-left: 5px solid #ffeeba;
        margin-top: 20px;
    }
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR & BRANDING
# ==========================================
with st.sidebar:
    st.markdown('<h1 style="font-size: 38px; margin-bottom: 0px;">Prakhar Avasthi</h1>', unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-top: -15px; margin-bottom: 15px; color: #DC3545; font-weight: 600; font-size: 16px; letter-spacing: 0.5px;'>Data Science & AI Professional</div>", 
        unsafe_allow_html=True
    )
    st.divider()
    
    st.link_button("🔗 LinkedIn", "https://linkedin.com/in/your-profile", use_container_width=True)
    st.link_button("🐙 GitHub", "https://github.com/your-github", use_container_width=True)
    st.link_button("📊 Kaggle", "https://kaggle.com/your-kaggle", use_container_width=True)
    
    st.divider()
    st.markdown("📧 prakharavasthi1999@gmail.com")
    
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.warning("⚠️ Groq API Key not found in Streamlit Secrets.")

# ==========================================
# 3. HELPER FUNCTIONS & LOADERS
# ==========================================
@st.cache_resource
def load_model():
    try:
        with open('cancer_model.pkl', 'rb') as file:
            model = pickle.load(file) 
        return model
    except Exception as e:
        st.error(f"Failed to load the ML model. Error: {e}")
        return None

cancer_model = load_model()

@st.cache_data(ttl=3600)
def fetch_available_models():
    """Fetches dynamic model list directly from Groq."""
    models_list = []
    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for m in data:
                    m_id = m.get("id")
                    if "whisper" not in m_id.lower():
                        models_list.append({"id": m_id, "display_name": f"Groq ({m_id})"})
        except Exception:
            pass
            
    # Safe Fallback to 8b-instant which is highly stable
    if not models_list:
        models_list.append({"id": "llama-3.1-8b-instant", "display_name": "Groq (llama-3.1-8b-instant) - Fallback"})
        
    return models_list

# ==========================================
# 4. MAIN UI (TABS)
# ==========================================
tab_diagnosis, tab_eda, tab_methodology = st.tabs([
    "🫁 Clinical Diagnostic Tool", 
    "📊 Exploratory Data Analysis", 
    "🧠 AI Methodology"
])

with tab_diagnosis:
    st.title("🫁 Lung Cancer Predictive Diagnostics & AI Triage")
    st.markdown("*An AI-assisted diagnostic prototype predicting lung cancer risk based on patient symptoms and demographics.*")
    
    st.markdown("""
    <div class="disclaimer">
    <b>⚖️ MANDATORY MEDICAL DISCLAIMER:</b> This application is a machine learning proof-of-concept developed by Prakhar Avasthi for portfolio purposes only. 
    It is <b>NOT</b> a certified medical device. The predictions and AI-generated text provided by this tool do not constitute medical advice, diagnosis, or treatment. 
    Always consult a qualified healthcare provider for medical concerns.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.header("1. Patient Clinical Profile")
        st.markdown("Enter the patient's symptomatic and clinical profile:")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 20, 100, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            smoking = st.radio("Smoking History?", ["No", "Yes"], horizontal=True)
            finger_discoloration = st.radio("Finger Discoloration?", ["No", "Yes"], horizontal=True)
            mental_stress = st.radio("Mental Stress?", ["No", "Yes"], horizontal=True)
            exposure_pollution = st.radio("Exposure to Pollution?", ["No", "Yes"], horizontal=True)
            
        with c2:
            long_term_illness = st.radio("Long Term Illness?", ["No", "Yes"], horizontal=True)
            energy_level = st.slider("Energy Level (Score)", 0.0, 100.0, 50.0)
            immune_weakness = st.radio("Immune Weakness?", ["No", "Yes"], horizontal=True)
            breathing_issue = st.radio("Breathing Issues?", ["No", "Yes"], horizontal=True)
            alcohol = st.radio("Alcohol Consumption?", ["No", "Yes"], horizontal=True)
            throat_discomfort = st.radio("Throat Discomfort?", ["No", "Yes"], horizontal=True)

        with c3:
            oxygen_saturation = st.slider("Oxygen Saturation (%)", 70.0, 100.0, 95.0)
            chest_tightness = st.radio("Chest Tightness?", ["No", "Yes"], horizontal=True)
            family_history = st.radio("Family History of Cancer?", ["No", "Yes"], horizontal=True)
            smoking_family = st.radio("Family History of Smoking?", ["No", "Yes"], horizontal=True)
            stress_immune = st.radio("Stress Immune Response?", ["No", "Yes"], horizontal=True)

        if st.button("Run Diagnostic ML Model", use_container_width=True, type="primary"):
            
            # PERFECT DATA MAPPING: Aligned with the Kaggle df.describe() which uses 0 and 1
            def map_bin(val): return 1 if val == "Yes" else 0
            
            # Save readable features for LLM context
            st.session_state['patient_features'] = {
                "Age": age, "Gender": gender, "Smoking": smoking,
                "Finger Discoloration": finger_discoloration, "Mental Stress": mental_stress,
                "Exposure to Pollution": exposure_pollution, "Long Term Illness": long_term_illness,
                "Energy Level": energy_level, "Immune Weakness": immune_weakness,
                "Breathing Issues": breathing_issue, "Alcohol Consumption": alcohol,
                "Throat Discomfort": throat_discomfort, "Oxygen Saturation": oxygen_saturation,
                "Chest Tightness": chest_tightness, "Family History": family_history,
                "Smoking Family History": smoking_family, "Stress Immune Response": stress_immune
            }
            
            if cancer_model:
                try:
                    # Map exactly to the DataFrame schema from the Kaggle notebook
                    input_data = {
                        "AGE": age,
                        "GENDER": 1 if gender == "Male" else 0,
                        "SMOKING": map_bin(smoking),
                        "FINGER_DISCOLORATION": map_bin(finger_discoloration),
                        "MENTAL_STRESS": map_bin(mental_stress),
                        "EXPOSURE_TO_POLLUTION": map_bin(exposure_pollution),
                        "LONG_TERM_ILLNESS": map_bin(long_term_illness),
                        "ENERGY_LEVEL": float(energy_level),
                        "IMMUNE_WEAKNESS": map_bin(immune_weakness),
                        "BREATHING_ISSUE": map_bin(breathing_issue),
                        "ALCOHOL_CONSUMPTION": map_bin(alcohol),
                        "THROAT_DISCOMFORT": map_bin(throat_discomfort),
                        "OXYGEN_SATURATION": float(oxygen_saturation),
                        "CHEST_TIGHTNESS": map_bin(chest_tightness),
                        "FAMILY_HISTORY": map_bin(family_history),
                        "SMOKING_FAMILY_HISTORY": map_bin(smoking_family),
                        "STRESS_IMMUNE": map_bin(stress_immune)
                    }
                    
                    input_df = pd.DataFrame([input_data])
                    
                    # Ensure columns are exactly as the model memorized them
                    if hasattr(cancer_model, "feature_names_in_"):
                        input_df = input_df[cancer_model.feature_names_in_]
                    
                    # Run Inference
                    prob = cancer_model.predict_proba(input_df)[0][1]
                    st.session_state['cancer_prob'] = prob
                    
                    st.success("Analysis Complete.")
                    if prob > 0.5:
                        st.error(f"⚠️ HIGH RISK DETECTED: {prob*100:.1f}% Probability of Malignancy")
                    else:
                        st.success(f"✅ LOW RISK DETECTED: {prob*100:.1f}% Probability of Malignancy")
                except Exception as e:
                    st.error(f"Inference Error: {e}")
            else:
                st.warning("⚠️ Model not found. Simulating output for demonstration...")

    with col2:
        st.header("2. AI Clinical Notes Generator")
        
        available_models = fetch_available_models()
        model_map = {m["display_name"]: m["id"] for m in available_models}
        
        selected_display_name = st.selectbox(
            "Select Target LLM Architecture", 
            options=list(model_map.keys()),
            help="Dynamically fetches your authorized models from Groq."
        )
        target_model_id = model_map[selected_display_name]
        
        st.markdown(f"The system will invoke **{target_model_id}** to generate a preliminary clinical summary.")
        
        if st.button("Generate AI Clinical Summary", use_container_width=True):
            if 'cancer_prob' not in st.session_state:
                st.warning("⚠️ Please successfully run the Diagnostic ML Model first so the AI has a probability score to analyze.")
            elif not GROQ_API_KEY:
                st.error("Groq API Ke
