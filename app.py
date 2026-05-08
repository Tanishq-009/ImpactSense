import streamlit as st
import numpy as np
import joblib


st.set_page_config(page_title="Earthquake Predictor", layout="centered")


st.markdown("""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


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

/* Subtitle */
.subtitle {
    text-align: center;
    color: #333;
    margin-bottom: 30px;
    font-size: 16px;
}

/* FIX LABEL VISIBILITY */
label {
    color: #1a237e !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* Remove wrapper */
[data-testid="stTextInput"] {
    background: transparent !important;
    border: none !important;
}

/* Input box */
[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #000 !important;
    border: 2px solid #ddd !important;
    border-radius: 8px !important;
    padding: 10px !important;
    outline: none !important;
    box-shadow: none !important;
    transition: 0.3s;
}

/* Focus effect */
[data-testid="stTextInput"] input:focus {
    border: 2px solid #ff416c !important;
    box-shadow: 0 0 8px rgba(255,65,108,0.3);
}

/* Remove red invalid border */
input:invalid {
    box-shadow: none !important;
}

/* Placeholder */
[data-testid="stTextInput"] input::placeholder {
    color: #888 !important;
}

/* Spacing */
[data-testid="stTextInput"] {
    margin-bottom: 15px;
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
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 15px rgba(255,65,108,0.3);
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

/* Alert */
.alert {
    margin-top: 15px;
    font-size: 14px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")


st.markdown('<div class="title">Earthquake Impact Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter seismic details to predict risk level</div>', unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    magnitude = st.text_input("Magnitude", placeholder="e.g. 6.5")
    depth = st.text_input("Depth (km)", placeholder="e.g. 10")
    cdi = st.text_input("CDI [Community Decimal Intensit]", placeholder="e.g. 5.5")

with col2:
    mmi = st.text_input("MMI [Modified Mercalli Intensity]", placeholder="e.g. 7")
    sig = st.text_input("Significance", placeholder="e.g. 500")


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

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)
        confidence = np.max(prob) * 100

        energy_display = f"{energy_approx:.2e}"

        if prediction[0] == 0:
            st.markdown(
                f'<div class="result low">LOW RISK ({confidence:.2f}%)<br>'
                f'Energy: {energy_display}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="alert" style="color:#2e7d32;">'
                'Situation is stable. Stay aware of surroundings.<br>'
                'Keep basic emergency supplies ready.'
                '</div>', unsafe_allow_html=True
            )

        elif prediction[0] == 1:
            st.markdown(
                f'<div class="result medium">MEDIUM RISK ({confidence:.2f}%)<br>'
                f'Energy: {energy_display}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="alert" style="color:#ef6c00;">'
                'Moderate risk detected. Stay alert.<br>'
                'Be prepared to move to a safer location.'
                '</div>', unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="result high">HIGH RISK ({confidence:.2f}%)<br>'
                f'Energy: {energy_display}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="alert" style="color:#c62828;">'
                'High risk detected. Take immediate precautions.<br>'
                'Move to an open and safe area away from structures.'
                '</div>', unsafe_allow_html=True
            )

    except:
        st.markdown('<div class="result high">Enter valid numeric values</div>', unsafe_allow_html=True)
