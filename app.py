"""
Streamlit Monolithic Interface for Lung Cancer Prediction, EDA, & Clinical AI Assistant.
"""
import os
import pickle
import numpy as np
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
    
    /* Increase Font Size for the Navigation Tabs */
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
    
    st.link_button("🔗 LinkedIn", "http://www.linkedin.com/in/prakhar-avasthi-35067a1bb", use_container_width=True)
    st.link_button("🐙 GitHub", "https://github.com/AvasthiPrakhar", use_container_width=True)
    st.link_button("📊 Kaggle", "https://www.kaggle.com/avasthiprakhar", use_container_width=True)
    
    st.divider()
    st.markdown("📧 prakharavasthi1999@gmail.com")
    
    # Safely load the Groq Key from Streamlit Secrets
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.warning("⚠️ Groq API Key not found in Streamlit Secrets.")

# ==========================================
# 3. LOAD ML MODEL (VIA PICKLE)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # Load your exported Kaggle pickle file
        with open('cancer_model.pkl', 'rb') as file:
            model = pickle.load(file) 
        return model
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

cancer_model = load_model()

# ==========================================
# 4. MAIN UI (TABS)
# ==========================================
tab_diagnosis, tab_eda, tab_methodology = st.tabs([
    "🫁 Clinical Diagnostic Tool", 
    "📊 Exploratory Data Analysis", 
    "🧠 AI Methodology"
])

# ------------------------------------------
# TAB 1: CLINICAL DIAGNOSTIC TOOL
# ------------------------------------------
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
        st.markdown("Enter the patient's symptomatic profile (1 = No, 2 = Yes):")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 20, 100, 50)
            smoking = st.radio("Smoking?", [1, 2], horizontal=True)
            yellow_fingers = st.radio("Yellow Fingers?", [1, 2], horizontal=True)
            anxiety = st.radio("Anxiety?", [1, 2], horizontal=True)
        with c2:
            peer_pressure = st.radio("Peer Pressure?", [1, 2], horizontal=True)
            chronic_disease = st.radio("Chronic Disease?", [1, 2], horizontal=True)
            fatigue = st.radio("Fatigue?", [1, 2], horizontal=True)
            allergy = st.radio("Allergy?", [1, 2], horizontal=True)
            wheezing = st.radio("Wheezing?", [1, 2], horizontal=True)
        with c3:
            alcohol = st.radio("Alcohol Consuming?", [1, 2], horizontal=True)
            coughing = st.radio("Coughing?", [1, 2], horizontal=True)
            shortness_of_breath = st.radio("Shortness of Breath?", [1, 2], horizontal=True)
            swallowing = st.radio("Swallowing Difficulty?", [1, 2], horizontal=True)
            chest_pain = st.radio("Chest Pain?", [1, 2], horizontal=True)

        if st.button("Run Diagnostic ML Model", use_container_width=True, type="primary"):
            # Prepare feature array in the exact order the Kaggle model expects
            gender_val = 1 if gender == "Male" else 0
            
            features = np.array([[
                gender_val, age, smoking, yellow_fingers, anxiety, peer_pressure, 
                chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, 
                shortness_of_breath, swallowing, chest_pain
            ]])
            
            # Save features for LLM context
            st.session_state['patient_features'] = {
                "Gender": gender, "Age": age, "Smoking": "Yes" if smoking==2 else "No",
                "Yellow Fingers": "Yes" if yellow_fingers==2 else "No",
                "Chronic Disease": "Yes" if chronic_disease==2 else "No",
                "Fatigue": "Yes" if fatigue==2 else "No",
                "Coughing": "Yes" if coughing==2 else "No",
                "Chest Pain": "Yes" if chest_pain==2 else "No"
            }

            if cancer_model:
                try:
                    prob = cancer_model.predict_proba(features)[0][1]
                    st.session_state['cancer_prob'] = prob
                    
                    st.success("Analysis Complete.")
                    if prob > 0.5:
                        st.error(f"⚠️ HIGH RISK DETECTED: {prob*100:.1f}% Probability of Malignancy")
                    else:
                        st.success(f"✅ LOW RISK DETECTED: {prob*100:.1f}% Probability of Malignancy")
                except Exception as e:
                    st.error(f"Inference Error: Please ensure your feature array matches the model's exact expected shape and order. Details: {e}")
            else:
                st.warning("⚠️ Model not found. Please ensure 'cancer_model.pkl' is uploaded to the repository.")

    with col2:
        st.header("2. AI Clinical Notes Generator")
        st.markdown("Uses **Groq LLaMA 3.3 70B** to generate a preliminary clinical summary based on the ML inference and patient symptoms.")
        
        if st.button("Generate AI Clinical Summary", use_container_width=True):
            if 'patient_features' not in st.session_state:
                st.warning("Please run the Diagnostic ML Model first.")
            elif not GROQ_API_KEY:
                st.error("Groq API Key is missing. Please add it to Streamlit Secrets.")
            else:
                with st.spinner("LLaMA 3.3 is analyzing patient profile..."):
                    try:
                        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.1)
                        
                        prompt = f"""
                        You are a highly professional Clinical AI Assistant.
                        
                        Patient Profile:
                        {st.session_state['patient_features']}
                        
                        Machine Learning Model Prediction (Cancer Probability): {st.session_state['cancer_prob']*100:.1f}%
                        
                        Write a 2-paragraph preliminary clinical summary. 
                        Paragraph 1: Summarize the patient's key risk factors (e.g., smoking, age, coughing) based on the profile.
                        Paragraph 2: State the ML model's probability score and recommend that the physician review the patient for further screening (e.g., CT Scan, biopsy).
                        
                        DO NOT diagnose the patient. Maintain a strictly objective, clinical tone suitable for a doctor's notes.
                        """
                        response = llm.invoke(prompt)
                        st.session_state['clinical_notes'] = response.content
                    except Exception as e:
                        st.error(f"LLM Error: {e}")
        
        if 'clinical_notes' in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state['clinical_notes'])

# ------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (KAGGLE)
# ------------------------------------------
with tab_eda:
    st.title("📊 Exploratory Data Analysis & Visualization")
    st.markdown("Below is the complete interactive Jupyter Notebook hosted on Kaggle, detailing the data cleaning, feature correlation, and visualization steps taken prior to model training.")
    st.divider()
    
    # Embed the Kaggle Notebook directly into Streamlit
    kaggle_iframe = """
    <iframe 
        src="https://www.kaggle.com/embed/avasthiprakhar/cancer-prediction-88-f1-90-acc-rf-cat-xgb-lgbm?kernelSessionId=228639734" 
        height="850" 
        style="margin: 0 auto; width: 100%; max-width: 1200px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" 
        frameborder="0" 
        scrolling="auto" 
        title="Cancer Prediction | 88% F1 | 90% Acc">
    </iframe>
    """
    components.html(kaggle_iframe, height=900)

# ------------------------------------------
# TAB 3: METHODOLOGY
# ------------------------------------------
with tab_methodology:
    st.title("🧠 Predictive Modeling Architecture")
    st.divider()
    
    met1, met2, met3 = st.columns(3)
    met1.metric("Overall Accuracy", "90.0%")
    met2.metric("F1-Score", "88.0%")
    met3.metric("False Negative Rate", "Optimized/Minimized")
    
    st.divider()
    
    st.write("### Model Training Overview")
    st.write("This application is backed by a robust ensemble machine learning pipeline developed by Prakhar Avasthi.")
    st.write("Multiple algorithms were benchmarked on a 5,000+ patient record dataset, including **Random Forest, CatBoost, XGBoost, and LightGBM**. Hyperparameter tuning was conducted utilizing `GridSearchCV` to optimize the model specifically for **Recall**, actively minimizing False Negatives.")
    st.info("💡 **Why optimize for Recall?** In medical diagnostics, a False Negative (missing a cancer diagnosis and sending a sick patient home) carries a drastically higher penalty than a False Positive (flagging a healthy patient for a secondary screening). The threshold was adjusted to ensure maximum sensitivity.")
