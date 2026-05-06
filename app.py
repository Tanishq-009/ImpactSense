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

/* ✍️ Inputs (GenZ style) */
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
