import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Earthquake Impact Predictor", layout="centered")

# ---- LOAD MODEL ----
model = joblib.load("rf_model.pkl")

# ---- CSS ----
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #5f0a87, #1a1a2e);
}
.block-container {
    max-width: 700px;
}
.title {
    text-align: center;
    color: white;
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #ddd;
    margin-bottom: 30px;
}
label {
    color: white !important;
    font-weight: 600 !important;
}
.stTextInput input {
    background-color: #2b2f38 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #666 !important;
    padding: 10px !important;
}
.stButton>button {
    border-radius: 12px;
    padding: 10px 20px;
    color: white;
    background: linear-gradient(90deg, #8a2be2, #4169e1);
    border: none;
}
.icon {
    text-align: center;
    font-size: 32px;
    margin-top: 20px;
}
.output {
    margin-top: 20px;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
}
.low { background: #d1fae5; color: #065f46; }
.medium { background: #fef3c7; color: #92400e; }
.high { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="icon">⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Earthquake Impact Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict earthquake risk based on seismic parameters</div>', unsafe_allow_html=True)

# ---- Inputs ----
magnitude = st.text_input("Magnitude", placeholder="Enter magnitude (e.g., 6.5)")
depth = st.text_input("Depth (km)", placeholder="Enter depth (e.g., 10)")
cdi = st.text_input("CDI", placeholder="Enter CDI (e.g., 5.5)")
mmi = st.text_input("MMI", placeholder="Enter MMI (e.g., 7)")
sig = st.text_input("Significance (SIG)", placeholder="Enter SIG (e.g., 500)")

# ---- Prediction ----
if st.button("Predict Risk"):

    try:
        # Convert input
        magnitude = float(magnitude)
        depth = float(depth)
        cdi = float(cdi)
        mmi = float(mmi)
        sig = float(sig)

        # Feature Engineering (IMPORTANT)
        mag_depth_interaction = magnitude * depth
        energy_approx = 10 ** (1.5 * magnitude)

        input_data = np.array([[
            magnitude,
            depth,
            cdi,
            mmi,
            sig,
            mag_depth_interaction,
            energy_approx
        ]])

        # Predict
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)
        confidence = np.max(prob) * 100

        # Output
        if prediction[0] == 0:
            st.markdown(f'<div class="output low">🟢 Low Risk ({confidence:.2f}%)</div>', unsafe_allow_html=True)
        elif prediction[0] == 1:
            st.markdown(f'<div class="output medium">🟡 Medium Risk ({confidence:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="output high">🔴 High Risk ({confidence:.2f}%)</div>', unsafe_allow_html=True)

    except:
        st.markdown('<div class="output high">⚠️ Please enter valid numeric values</div>', unsafe_allow_html=True)