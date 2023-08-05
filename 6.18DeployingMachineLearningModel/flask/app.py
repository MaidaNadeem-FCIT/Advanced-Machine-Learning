import pandas as pd
import joblib
from flask import Flask, render_template, request

#Object of Flask
app = Flask(__name__)

#Load Model
model = joblib.load("C:/Users/ASCC/Documents/GitHub/Advanced-Machine-Learning/6.18DeployingMachineLearningModel/lr_model_adv.pkl")

#Decorator
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    #Getting Values From Html Form
    tv, radio, newspaper = [float(x) for x in request.form.values()]
    #Format Input User data into a Dataframe
    new_data = pd.DataFrame({'TV':tv, "radio":radio, "newspaper":newspaper}, index=[0])
    result = model.predict(new_data)
    return render_template("index.html", prediction_text="Predicted Sales (in Thousand units) is {}".format(result))

if __name__ == '__main__':
    app.run(debug=True)
