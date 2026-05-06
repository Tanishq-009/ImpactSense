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
# UI STYLING
# ==============================
st.markdown("""
<style>

/* Background */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f7fa, #fce4ec) !important;
}

/* Container */
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

/* Input wrapper remove */
[data-testid="stTextInput"] {
    background: transparent !important;
    border: none !important;
}

/* Input box */
[data-testid="stTextInput"] input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #ccc !important;
    border-radius: 4px !important;
    padding: 10px !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Focus */
[data-testid="stTextInput"] input:focus {
    border: 2px solid #7b2ff7 !important;
    box-shadow: none !important;
}

/* Remove red invalid */
input:invalid {
    box-shadow: none !important;
}

/* Placeholder */
[data-testid="stTextInput"] input::placeholder {
    color: #888 !important;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    padding: 14px;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border: none;
}

/* Result */
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}

.low { background: #e8f5e9; color: #2e7d32; }
.medium { background: #fff8e1; color: #ef6c00; }
.high { background: #ffebee; color: #c62828; }

/* Safety message */
.alert {
    margin-top: 20px;
    font-size: 14px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# HEADER
# ==============================
st.markdown('<div class="title">Earthquake Impact Predictor</div>', unsafe_allow_html=True)
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
# PREDICTION
# ==============================
if st.button("Predict Impact"):
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

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)
        confidence = np.max(prob) * 100

        energy_display = f"{energy_approx:.2e}"

        # Output
        if prediction[0] == 0:
            st.markdown(
                f'<div class="result low">LOW RISK ({confidence:.2f}%)<br>'
                f'Energy: {energy_display}</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
            <div class="alert" style="color:#2e7d32;">
            Situation is stable. Stay aware of surroundings.<br>
            Keep basic emergency supplies ready.
            </div>
            """, unsafe_allow_html=True)

        elif prediction[0] == 1:
            st.markdown(
                f'<div class="result medium">MEDIUM RISK ({confidence:.2f}%)<br>'
                f'Energy: {energy_display}</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
            <div class="alert" style="color:#ef6c00;">
            Moderate risk detected. Stay alert.<br>
            Be prepared to move to a safer location.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(
                f'<div class="result high">HIGH RISK ({confidence:.2f}%)<br>'
                f'Energy: {energy_display}</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
            <div class="alert" style="color:#c62828;">
            High risk detected. Take immediate precautions.<br>
            Move to an open and safe area away from structures.
            </div>
            """, unsafe_allow_html=True)

    except:
        st.markdown('<div class="result high">Enter valid numeric values</div>', unsafe_allow_html=True)
