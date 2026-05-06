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
# UI DESIGN (CLEAN + FIXED INPUT BUG)
# ==============================
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f7fa, #fce4ec) !important;
}

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

/* Inputs FIXED */
.stTextInput input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #ccc !important;
    border-radius: 12px !important;
    padding: 12px !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Remove weird focus box */
.stTextInput input:focus {
    border: 2px solid #7b2ff7 !important;
    box-shadow: 0 0 8px rgba(123,47,247,0.3) !important;
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

/* Result box */
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

.low { background: #e8f5e9; color: #2e7d32; }
.medium { background: #fff8e1; color: #ef6c00; }
.high { background: #ffebee; color: #c62828; }

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

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
    magnitude = st.text_input("Magnitude", placeholder="e.g. 6.5")
    depth = st.text_input("Depth (km)", placeholder="e.g. 10")
    cdi = st.text_input("CDI", placeholder="e.g. 5.5")

with col2:
    mmi = st.text_input("MMI", placeholder="e.g. 7")
    sig = st.text_input("Significance", placeholder="e.g. 500")

# ==============================
# PREDICT
# ==============================
if st.button("Predict Impact"):
    try:
        magnitude = float(magnitude)
        depth = float(depth)
        cdi = float(cdi)
        mmi = float(mmi)
        sig = float(sig)

        # SAME FEATURE ENGINEERING AS TRAINING
        mag_depth_interaction = magnitude * depth
        energy_approx = 10 ** (1.5 * magnitude)

        input_data = np.array([[ 
            magnitude, depth, cdi, mmi, sig,
            mag_depth_interaction, energy_approx
        ]])

        # ✅ APPLY SCALING (IMPORTANT FIX)
        input_scaled = scaler.transform(input_data)

        # ✅ PREDICT
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)
        confidence = np.max(prob) * 100

        # EXTRA INFO
        energy_display = f"{energy_approx:.2e}"
        depth_impact = magnitude / (depth + 1)

        # RESULT OUTPUT
        if prediction[0] == 0:
            st.markdown(
                f'<div class="result low"> LOW RISK ({confidence:.2f}%)<br>'
                f' Energy: {energy_display}<br>'
                f' Depth Impact: {depth_impact:.2f}</div>',
                unsafe_allow_html=True
            )
        elif prediction[0] == 1:
            st.markdown(
                f'<div class="result medium"> MEDIUM RISK ({confidence:.2f}%)<br>'
                f' Energy: {energy_display}<br>'
                f' Depth Impact: {depth_impact:.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result high"> HIGH RISK ({confidence:.2f}%)<br>'
                f'Energy: {energy_display}<br>'
                f' Depth Impact: {depth_impact:.2f}</div>',
                unsafe_allow_html=True
            )

        # DEBUG (remove later)
        st.write("Prediction:", prediction)
        st.write("Probabilities:", prob)

    except:
        st.markdown('<div class="result high">⚠️ Enter valid numeric values</div>', unsafe_allow_html=True)
