import pandas as pd
import joblib
import streamlit as st

# loading model
model = joblib.load("practical_model_adv.pkl")

#UI Elements to get Imput From User
st.title("Predicting Titanic Survivor")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1)
sex = st.selectbox("Gender", ['male', 'female'])
embark = st.selectbox("Embarked From", ['Cherbourg','Southampton','Queenstown'])
if embark =='Cherbourg':
    embark='C'
elif embark =='Southampton':
    embark='S'
else:
    embark='Q'
alone= st.selectbox("Travelling Alone", ["Yes", "No"])
if alone =='Yes':
    alone=1
else:
    alone=0

#Format Input User data into a Dataframe
new_data = pd.DataFrame({'pclass':pclass, "age":age, "sex":sex, "embarked":embark, "Travel_alone":alone}, index=[0])

result = ""
if st.button("Predict"):
    result = model.predict(new_data)
    if result == 1:
        result = "Survived"
    else:
        result= "Not Survived"

    st.subheader("Predictd Status of the Person")
    st.subheader(result)
else:
    st.subheader("Enter the required Data or Press Predict")