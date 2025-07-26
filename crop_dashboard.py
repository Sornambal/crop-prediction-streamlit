
import streamlit as st
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Smart Crop Predictor 🌱", layout="centered")
st.title("🌾 Smart Crop Recommendation System")

st.markdown("### 📋 Input Details")

# Mandatory Inputs
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=60.0, step=0.1, format="%.2f")
ph = st.number_input("🧪 pH Level", min_value=0.0, max_value=14.0, step=0.1, format="%.2f")
rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1, format="%.2f")

# Optional Inputs
with st.expander("➕ Optional Inputs"):
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f")
    N = st.number_input("🌿 Nitrogen (N)", min_value=0.0, max_value=140.0, step=0.1, format="%.2f")
    P = st.number_input("🪴 Phosphorus (P)", min_value=5.0, max_value=145.0, step=0.1, format="%.2f")
    K = st.number_input("🌱 Potassium (K)", min_value=5.0, max_value=205.0, step=0.1, format="%.2f")

if st.button("Predict Best Crop 🌾"):
    # If optional inputs left as 0, handle accordingly
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    crop_name = label_encoder.inverse_transform([prediction])[0]

    st.success(f"✅ Recommended Crop: **{crop_name.upper()}**")
    st.balloons()
