import streamlit as st
import numpy as np
import pickle

# Load model and encoder
model = pickle.load(open("crop_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# App title
st.title("🌾 Crop Recommendation System")

st.write("### Enter the details to predict the suitable crop:")

# User input features
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

if st.button("Predict Crop"):
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction_encoded = model.predict(user_input)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    st.success(f"✅ Recommended Crop: **{prediction_label.upper()}**")
