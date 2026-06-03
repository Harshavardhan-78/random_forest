import streamlit as st
import pandas as pd
import pickle

# =========================
# Load Model & Columns
# =========================

model = pickle.load(open("models/random_forest_classifier.pkl", "rb"))

model_columns = pickle.load(open("models/model_columns_classifier.pkl", "rb"))

# =========================
# Streamlit UI
# =========================

st.title("🚗 Car Price Category Prediction")

st.write("Predict whether a car price is Low, Medium, or High")

# =========================
# Numerical Inputs
# =========================

year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
mileage = st.number_input("Mileage", min_value=0.0, value=20.0)
engine = st.number_input("Engine CC", min_value=0.0, value=1200.0)
max_power = st.number_input("Max Power", min_value=0.0, value=80.0)
torque = st.number_input("Torque", min_value=0.0, value=190.0)
seats = st.number_input("Seats", min_value=1, max_value=10, value=5)

# =========================
# Categorical Inputs
# =========================

fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol", "LPG", "CNG"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

# =========================
# Input Data
# =========================

input_data = {
    'year': year,
    'km_driven': km_driven,
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power,
    'torque': torque,
    'seats': seats,
    'fuel_Diesel': 1 if fuel == 'Diesel' else 0,
    'fuel_LPG': 1 if fuel == 'LPG' else 0,
    'fuel_Petrol': 1 if fuel == 'Petrol' else 0,
    'seller_type_Individual': 1 if seller_type == 'Individual' else 0,
    'seller_type_Trustmark Dealer': 1 if seller_type == 'Trustmark Dealer' else 0,
    'transmission_Manual': 1 if transmission == 'Manual' else 0,
    'owner_Second Owner': 1 if owner == 'Second Owner' else 0,
    'owner_Third Owner': 1 if owner == 'Third Owner' else 0,
    'owner_Fourth & Above Owner': 1 if owner == 'Fourth & Above Owner' else 0,
    'owner_Test Drive Car': 1 if owner == 'Test Drive Car' else 0
}

# =========================
# Create DataFrame & Align Columns
# =========================

input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(
    columns=model_columns,
    fill_value=0
)

# =========================
# Prediction
# =========================

if st.button("Predict Category"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price Category: {prediction}")

# =========================
# Footer
# =========================

st.caption("Built using Streamlit & Scikit-Learn")