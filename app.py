import streamlit as st
import numpy as np
import pickle
import requests

# 🌐 Weather API Setup
API_KEY = "72b890e0979d914834509b2506b64de9"  # 🔑 Replace with your OpenWeatherMap API key

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data["cod"] == 200:
        temp = data["main"]["temp"]
        humidity = float(data["main"]["humidity"])
        rainfall = float(data["rain"]["1h"]) if "rain" in data and "1h" in data["rain"] else 0.0
        return temp, humidity, rainfall
    else:
        st.warning(f"❌ Could not fetch weather for '{city}'. Error: {data.get('message', 'Unknown error')}")
        return None, None, None

# 🌾 Streamlit UI
st.set_page_config(page_title="Smart Crop Recommendation System", layout="centered")
st.title("🌾 Crop Recommendation System")

st.markdown("📍 Auto-detect **weather conditions** from your city input")
st.markdown("🧪 Please **manually enter soil nutrients and pH** based on your soil report.")

city = st.text_input("Enter City Name with Country Code (e.g., 'Coimbatore,IN')", value="Coimbatore,IN")

temperature, humidity, rainfall = None, None, None

if city:
    temperature, humidity, rainfall = get_weather(city)
    if temperature is not None:
        st.success("✅ Weather data autofilled successfully.")
    else:
        st.info("ℹ️ You can manually fill the values if auto-fetch fails.")

# 🧪 Soil and Environmental Inputs
st.markdown("### 🔢 Enter the soil values below:")
st.markdown("*You can get these from a soil testing report.*")

N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# 🌡️ Auto-filled or manual climate data
st.markdown("### 🌦️ Auto-filled Weather Data (Editable if needed):")
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=float(temperature) if temperature is not None else 25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(humidity) if humidity is not None else 70.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=float(rainfall) if rainfall is not None else 100.0)


# Load model and encoder
model = pickle.load(open("crop_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# 🔮 Predict Button
if st.button("Predict Crop"):
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction_encoded = model.predict(user_input)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    st.success(f"✅ Recommended Crop: **{prediction_label.upper()}**")
