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
# BRIGHT UI CSS
# ==============================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f6f9fc, #e3f2fd);
    font-family: 'Segoe UI', sans-serif;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #1a237e;
}

.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 25px;
}

/* Labels */
label {
    color: #333 !important;
    font-weight: 600 !important;
}

/* Input fields */
.stTextInput input {
    background-color: #f9fafc !important;
    color: #000 !important;
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
    padding: 10px !important;
}

/* Placeholder visibility */
.stTextInput input::placeholder {
    color: #888 !important;
    opacity: 1 !important;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
    background: linear-gradient(90deg, #2196f3, #21cbf3);
    color: white;
    border: none;
}

/* Result box */
.result {
    margin-top: 20px;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}

/* Colors */
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
# INPUT CARD
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    magnitude = st.text_input("Magnitude", placeholder="e.g. 6.5")
    depth = st.text_input("Depth (km)", placeholder="e.g. 10")
    cdi = st.text_input("CDI", placeholder="e.g. 5.5")

with col2:
    mmi = st.text_input("MMI", placeholder="e.g. 7")
    sig = st.text_input("Significance", placeholder="e.g. 500")

st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict Risk"):
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
