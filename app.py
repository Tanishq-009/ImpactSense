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
# FULL UI FIX + DESIGN
# ==============================
st.markdown("""
<style>

/* BACKGROUND */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f7fa, #fce4ec) !important;
}

/* REMOVE GREY */
[data-testid="stAppViewContainer"], .block-container {
    background: transparent !important;
}

/* TITLE */
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

/* LABELS */
label {
    color: #333 !important;
    font-weight: 600 !important;
}

/* INPUT FIX (NO GHOST BOX) */
.stTextInput > div > div > input {
    background-color: #ffffff !important;
    color: #000 !important;
    border: 2px solid #ddd !important;
    border-radius: 12px !important;
    padding: 12px !important;
    box-shadow: none !important;
    outline: none !important;
    transition: all 0.2s ease-in-out;
}

/* REMOVE INNER LAYER */
.stTextInput > div {
    background: transparent !important;
    border: none !important;
}

/* FOCUS EFFECT */
.stTextInput > div > div > input:focus {
    border: 2px solid #ff416c !important;
    box-shadow: 0 0 6px rgba(255,65,108,0.3) !important;
}

/* BUTTON */
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

/* BUTTON HOVER */
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px rgba(255,75,43,0.4);
}

/* RESULT BOX */
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}

/* COLORS */
.low { background: #e8f5e9; color: #2e7d32; }
.medium { background: #fff8e1; color: #ef6c00; }
.high { background: #ffebee; color: #c62828; }

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
if st.button("🚀 Predict Impact"):
    try:
        magnitude = float(magnitude)
        depth = float(depth)
        cdi = float(cdi)
        mmi = float(mmi)
        sig = float(sig)

        # Feature Engineering
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

        # INTERPRETATION
        if prediction[0] == 0:
            risk = "LOW"
            color_class = "low"
            advice = "Minimal damage expected. Stay aware but no immediate danger."
        elif prediction[0] == 1:
            risk = "MEDIUM"
            color_class = "medium"
            advice = "Moderate impact possible. Stay alert and follow safety guidelines."
        else:
            risk = "HIGH"
            color_class = "high"
            advice = "Severe impact likely. Immediate precautions required."

        # OUTPUT
        st.markdown(f"""
        <div class="result {color_class}">
            🌍 <b>{risk} RISK</b><br><br>
            📊 Confidence: {confidence:.2f}%<br>
            ⚡ Energy: {energy_approx:.2e}<br>
            📌 Depth Impact: {mag_depth_interaction:.2f}<br><br>
            🧠 <i>{advice}</i>
        </div>
        """, unsafe_allow_html=True)

    except:
        st.markdown(
            '<div class="result high">⚠️ Please enter valid numeric values</div>',
            unsafe_allow_html=True
        )
