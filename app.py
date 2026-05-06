import streamlit as st
import numpy as np
import joblib

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Earthquake Predictor", layout="centered")

# ==============================
# HIDE STREAMLIT UI
# ==============================
st.markdown("""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# FIX BACKGROUND + THEME OVERRIDE
# ==============================
st.markdown("""
<style>

/* FORCE FULL BACKGROUND */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f7fa, #fce4ec) !important;
}

/* Remove grey overlay */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* Remove container background */
.block-container {
    background: transparent !important;
    padding-top: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b2b, #ff416c, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}

/* Labels */
label {
    color: #333 !important;
    font-weight: 600 !important;
}

/* Inputs */
.stTextInput input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #ddd !important;
    border-radius: 12px !important;
    padding: 12px !important;
    transition: 0.3s;
}

/* Input focus */
.stTextInput input:focus {
    border: 2px solid #7b2ff7 !important;
    box-shadow: 0 0 10px rgba(123,47,247,0.3);
}

/* Placeholder */
.stTextInput input::placeholder {
    color: #999 !important;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 14px;
    padding: 14px;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(90deg, #ff416c, #ff4b2b) !important;
    color: white !important;
    border: none;
}

/* Hover */
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px rgba(255,75,43,0.4);
}

/* Result box */
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

/* Colors */
.low {
    background: #e8f5e9;
    color: #2e7d32;
}
.medium {
    background: #fff8e1;
    color: #ef6c00;
}
.high {
    background: #ffebee;
    color: #c62828;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("rf_model.pkl")

# ==============================
# HEADER
# ==============================
st.markdown('<div class="title">🌍 Earthquake Impact Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter seismic details to predict risk level</div>', unsafe_allow_html=True)

# ==============================
# INPUTS
# ==============================
col1, col2 = st.columns(2)

with col1:
    magnitude = st.text_input("📊 Magnitude", placeholder="e.g. 6.5")
    depth = st.text_input("🌊 Depth (km)", placeholder="e.g. 10")
    cdi = st.text_input("📍 CDI", placeholder="e.g. 5.5")

with col2:
    mmi = st.text_input("📶 MMI", placeholder="e.g. 7")
    sig = st.text_input("⚡ Significance", placeholder="e.g. 500")

# ==============================
# PREDICT
# ==============================
if st.button("🚀 Predict Impact"):
    try:
        magnitude = float(magnitude)
        depth = float(depth)
        cdi = float(cdi)
        mmi = float(mmi)
        sig = float(sig)

        mag_depth_interaction = magnitude * depth
        energy_approx = 10 ** (1.5 * magnitude)

        input_data = np.array([[ 
            magnitude, depth, cdi, mmi, sig,
            mag_depth_interaction, energy_approx
        ]])

        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)
        confidence = np.max(prob) * 100

        if prediction[0] == 0:
            st.markdown(f'<div class="result low">🟢 LOW RISK ({confidence:.2f}%)</div>', unsafe_allow_html=True)
        elif prediction[0] == 1:
            st.markdown(f'<div class="result medium">🟡 MEDIUM RISK ({confidence:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result high">🔴 HIGH RISK ({confidence:.2f}%)</div>', unsafe_allow_html=True)

    except:
        st.markdown('<div class="result high">⚠️ Enter valid numeric values</div>', unsafe_allow_html=True)
