import pandas as pd
import joblib
import streamlit as st

# loading model
model = joblib.load("C:/Users/ASCC/Documents/GitHub/Advanced-Machine-Learning/6.18DeployingMachineLearningModel/lr_model_adv.pkl")

#UI Elements to get Imput From User
st.title("Predicting Advertising Budget")
tv = st.number_input("TV", step=1)
radio = st.number_input("Radio", step=1)
news = st.number_input("Newspaper", step=1)

#Format Input User data into a Dataframe
new_data = pd.DataFrame({'TV':tv, "radio":radio, "newspaper":news}, index=[0])

result = ""
if st.button("Predict"):
    result = model.predict(new_data)
    st.subheader("Predictd Sales of Item (in Thousands Units)")
    st.subheader(result)
else:
    st.subheader("Enter a Budget and Press Predict.")