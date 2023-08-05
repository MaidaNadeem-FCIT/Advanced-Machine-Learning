import pandas as pd
import joblib
from flask import Flask, jsonify, request

#Object of Flask
app = Flask(__name__)

#Load Model
model = joblib.load("C:/Users/ASCC/Documents/GitHub/Advanced-Machine-Learning/6.18DeployingMachineLearningModel/lr_model_adv.pkl")

@app.route('/predict', methods = ['POST'])
def predict():
    #Input User data 
    new_data = request.json
    df = pd.DataFrame(new_data)
    prediction = model.predict(df)
    return jsonify({"Prediction" : float(prediction)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=12345)
