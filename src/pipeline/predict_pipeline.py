import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self): #constructor for the predict pipeline class
        pass

    def predict(self, features): #method to predict the target variable for the given input data
        try:

            model_path = 'artifacts/model.pkl' #path to the trained model
            preprocessor_path = 'artifacts/preprocessor.pkl' #path to the preprocessor object
            model = load_object(file_path=model_path) #loading the trained model
            preprocessor = load_object(file_path=preprocessor_path) #loading the preprocessor object
            data_scaled = preprocessor.transform(features) #scaling the input data using the preprocessor
            pred = model.predict(data_scaled) #predicting the target variable using the trained model
            return pred #returning the predicted target variable  
        except Exception as e:
            raise CustomException(e, sys) #raising a custom exception with the error message and the system information 

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_data_frame(self): #method to convert the custom data into a pandas dataframe
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict) #returning the custom data as a pandas dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
