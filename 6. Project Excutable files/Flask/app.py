import pickle
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import xgboost

model = pickle.load(open('bbyp.pkl','rb'))
app=Flask(__name__)

@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/details', methods=["GET"])
def show_form():
    return render_template('details.html')

@app.route('/predict',methods=["POST","GET"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_values = [np.array(input_features) ]
    print(features_values)
    
    col = ['clonesize','honeybee','bumbles','andrena','osmia','MaxOfUpperTRange', 
           'MinOfUpperTRange', 'AverageOfUpperTRange', 'MaxOfLowerTRange','MinOfLowerTRange', 
           'AverageOfLowerTRange','RainingDays', 'AverageRainingDays', 
           'fruitset','fruitmass','seeds' ]
              
   
    df = pd.DataFrame(features_values, columns= col)
    
    prediction = model.predict(df)
    print(prediction[0])
    rounded_value = round(prediction[0], 2)
    text="Hence,based on calculation, the predicted BlueBerry Yield is: "
    
    return render_template('predict.html', prediction_text=text + str(rounded_value))




if __name__ == "__main__":
    app.run(debug=False,port= 5000)