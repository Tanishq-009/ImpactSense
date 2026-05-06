import streamlit as st
import numpy as np
import joblib

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Earthquake Predictor", layout="centered")

# ==============================
# HIDE STREAMLIT DEFAULT UI
# ==============================
st.markdown("""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# GEN-Z UI CSS
# ==============================
st.markdown("""
<style>

/* 🌈 Background */
.stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f7fa, #fce4ec);
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}

/* 🧠 Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b2b, #ff416c, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 16px;
    margin-bottom: 30px;
}

/* 🏷 Labels */
label {
    color: #333 !important;
    font-weight: 600 !important;
    font-size: 14px;
}

/* ✍️ Inputs */
.stTextInput input {
    background: rgba(255,255,255,0.7) !important;
    border: 2px solid transparent !important;
    border-radius: 12px !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(6px);
}

/* Focus glow */
.stTextInput input:focus {
    border: 2px solid #7b2ff7 !important;
    box-shadow: 0 0 10px rgba(123,47,247,0.4);
}

/* Placeholder */
.stTextInput input::placeholder {
    color: #999 !important;
}

/* 🔥 Button */
.stButton>button {
    width: 100%;
    border-radius: 14px;
    padding: 14px;
    font-size: 17px;
    font-weight: 600;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border: none;
    transition: all 0.3s ease;
}

/* Button hover */
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px rgba(255,75,43,0.4);
}

/* 🎯 Result card */
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    animation: fadeIn 0.5s ease-in-out;
}

/* 🎨 Result colors */
.low {
    background: linear-gradient(135deg, #a8ff78, #78ffd6);
    color: #064e3b;
}
.medium {
    background: linear-gradient(135deg, #ffe259, #ffa751);
    color: #7c2d12;
}
.high {
    background: linear-gradient(135deg, #ff6a6a, #ff3d3d);
    color: white;
}

/* ✨ Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
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
# PREDICTION
# ==============================
if st.button("Predict Impact"):
    try:
        magnitude = float(magnitude)
        depth = float(depth)
        cdi = float(cdi)
        mmi = float(mmi)
        sig = float(sig)

        # Feature engineering
        mag_depth_interaction = magnitude * depth
        energy_approx = 10 ** (1.5 * magnitude)

        input_data = np.array([[ 
            magnitude, depth, cdi, mmi, sig,
            mag_depth_interaction, energy_approx
        ]])

        # Prediction
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)
        confidence = np.max(prob) * 100

        # Output
        if prediction[0] == 0:
            st.markdown(
                f'<div class="result low">🟢 LOW RISK<br>{confidence:.2f}% Confidence</div>',
                unsafe_allow_html=True
            )
        elif prediction[0] == 1:
            st.markdown(
                f'<div class="result medium">🟡 MEDIUM RISK<br>{confidence:.2f}% Confidence</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result high">🔴 HIGH RISK<br>{confidence:.2f}% Confidence</div>',
                unsafe_allow_html=True
            )

    except:
        st.markdown(
            '<div class="result high">⚠️ Enter valid numeric values</div>',
            unsafe_allow_html=True
        )
