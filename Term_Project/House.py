import pandas as pd
import numpy as np
import joblib
import streamlit as st

# loading model
model = joblib.load("house_prediction_model.pkl")

#UI Elements to get Imput From User
st.title("Predicting House Prices in Lahore.")
st.sidebar.header("Input Features")

predefined_addresses = [
    'Al Rehman Garden', 'Bahria Town',
    'DHA', 'Faisal Town',
    'Green City', 'Johar Town', 
    'LDA Avenue', 'Model Town',
    'New Garden Town', 'Tariq Garden',
    'Wapda Town'
]

bed = st.sidebar.slider("Number of Bedrooms", 2, 7, 3)
bath = st.sidebar.slider("Number of Bathrooms", 2, 7, 2)
area = st.sidebar.number_input("Area (Marlas)", 3.0, 1000.0, 10.0)
add = st.sidebar.selectbox("Select an Address", predefined_addresses)

#Format Input User data into a Dataframe
new_data = pd.DataFrame({'Address':add, "Bedroom":bed, "Bathroom":bath, "Area(Marlas)":area}, index=[0])

result = ""
if st.button("Predict"):
    result = model.predict(new_data)
    formatted_result = f"{result[0]:.2f}"
    st.subheader(f"The predicted price House at {add} is ${formatted_result} Crore.")
else:
     st.warning("Please enter an address to make a prediction.")