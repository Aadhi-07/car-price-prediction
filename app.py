import streamlit as st
import numpy as np
import pickle

# Load model & encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Used Car Price Prediction")

# ---------------- INPUTS ----------------
manufacturer = st.text_input("Manufacturer")
model_name = st.text_input("Model")
category = st.text_input("Category")
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "CNG", "LPG"])
gear = st.selectbox("Gear Box", ["Manual", "Automatic"])
drive = st.selectbox("Drive Wheels", ["Front", "Rear", "4x4"])
doors = st.selectbox("Doors", ["2", "4", ">5"])
wheel = st.selectbox("Wheel", ["Left wheel", "Right wheel"])
color = st.text_input("Color")

leather = st.selectbox("Leather Interior", ["Yes", "No"])
engine = st.number_input("Engine Volume (e.g., 2.0)", min_value=0.5)
turbo = st.selectbox("Turbo", ["No", "Yes"])
mileage = st.number_input("Mileage (km)", min_value=0)
cylinders = st.number_input("Cylinders", min_value=1)
airbags = st.number_input("Airbags", min_value=0)
car_age = st.slider("Car Age (years)", 0, 40)

# ---------------- ENCODING ----------------
def safe_encode(col, val):
    try:
        return encoders[col].transform([val])[0]
    except:
        return 0

leather = 1 if leather == "Yes" else 0
turbo = 1 if turbo == "Yes" else 0

features = np.array([[
    safe_encode("Manufacturer", manufacturer),
    safe_encode("Model", model_name),
    safe_encode("Category", category),
    leather,
    safe_encode("Fuel type", fuel),
    engine,
    mileage,
    cylinders,
    safe_encode("Gear box type", gear),
    safe_encode("Drive wheels", drive),
    safe_encode("Doors", doors),
    safe_encode("Wheel", wheel),
    safe_encode("Color", color),
    airbags,
    turbo,
    car_age
]])

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):
    log_price = model.predict(features)[0]
    price = np.expm1(log_price)
    st.success(f"💰 Estimated Car Price: ${price:,.2f}")
