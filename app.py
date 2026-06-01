import pickle
import numpy as np
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

#Route for home page

@app.route('/') #decorator for the home page route  

def index():
    return render_template('index.html') #rendering the index.html file when the home page is accessed


# @app.route('/predictdata', methods=['GET', 'POST'])
@app.route('/predict_datapoint', methods=['GET', 'POST']) #used to predict the target variable for the given input data
def predict_datapoint():
    if request.method == 'GET': #checking if the request method is POST
        return render_template('home.html') #rendering the home.html file if the request method is GET
    else:
        data = custum_data = CustomData(
            gender = request.form.get('gender'), #getting the gender from the form data
            race_ethnicity= request.form.get('ethnicity'), #getting the race race_ethnicity
            parental_level_of_education = request.form.get('parental_level_of_education'), #getting the parental level of education from the form data
            lunch = request.form.get('lunch'), #getting the lunch from the form data
            test_preparation_course = request.form.get('test_preparation_course'), #getting the test preparation course from the form data
            reading_score = int(request.form.get('reading_score')), #getting the reading score from the form data and converting it to an integer
            writing_score = int(request.form.get('writing_score')) #getting the writing score from the form data and converting it to an integer
        )

        print(request.form) #printing the form data to check if the data is correctly received from the form

        pred_df = data.get_data_as_data_frame() #converting the custom data into a pandas dataframe
        print(pred_df) #printing the pandas dataframe to check if the data is correctly converted into a dataframe

        predict_pipeline = PredictPipeline() #creating an object of the PredictPipeline class
        results = predict_pipeline.predict(pred_df) #predicting the target variable for the given input data using the predict method of the PredictPipeline class
        return render_template('home.html', results=results[0]) #rendering the home.html file and passing the predicted result to the template to display it on the web page
    
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True) #running the flask application in debug mode  