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
# CUSTOM CSS (GLASS UI)
# ==============================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);
    font-family: 'Segoe UI', sans-serif;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #ccc;
    margin-bottom: 30px;
}

.stSlider > div {
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border: none;
}

.result {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
}

.low { background: rgba(0,255,150,0.2); color: #00ffae; }
.medium { background: rgba(255,200,0,0.2); color: #ffd700; }
.high { background: rgba(255,0,0,0.2); color: #ff4c4c; }

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
st.markdown('<div class="subtitle">AI-powered seismic risk analysis</div>', unsafe_allow_html=True)

# ==============================
# INPUT CARD
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    magnitude = st.slider("Magnitude", 0.0, 10.0, 5.5)
    depth = st.slider("Depth (km)", 0, 700, 50)

with col2:
    cdi = st.slider("CDI", 0.0, 10.0, 5.0)
    mmi = st.slider("MMI", 0.0, 10.0, 6.0)
    sig = st.slider("Significance", 0, 1000, 500)

st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PREDICT BUTTON
# ==============================
if st.button("🚀 Predict Impact"):

    # Feature engineering
    mag_depth_interaction = magnitude * depth
    energy_approx = 10 ** (1.5 * magnitude)

    input_data = np.array([[ 
        magnitude, depth, cdi, mmi, sig,
        mag_depth_interaction, energy_approx
    ]])

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)
    confidence = np.max(prob) * 100

    # ==============================
    # RESULT CARD
    # ==============================
    if prediction[0] == 0:
        st.markdown(f'<div class="result low">🟢 LOW RISK<br>{confidence:.2f}% Confidence</div>', unsafe_allow_html=True)
    elif prediction[0] == 1:
        st.markdown(f'<div class="result medium">🟡 MEDIUM RISK<br>{confidence:.2f}% Confidence</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result high">🔴 HIGH RISK<br>{confidence:.2f}% Confidence</div>', unsafe_allow_html=True)

    # ==============================
    # CONFIDENCE BAR
    # ==============================
    st.progress(int(confidence))
