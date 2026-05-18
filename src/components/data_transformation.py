import sys 
from dataclasses import dataclass
from typing import Self
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # for applying different transformations to different columns
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.pipeline import Pipeline # for creating a pipeline of transformations
from sklearn.preprocessing import OneHotEncoder, StandardScaler # for encoding categorical variables and scaling numerical variables
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object # for saving the preprocessor object to the given path

@dataclass # for storing configuration related to data transformation
class DataTransformationConfig: # for storing configuration related to data transformation
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')# path to save the preprocessor object and artifacts folder is where we save all the artifacts related to the project

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # creating an instance of the DataTransformationConfig class to access the configuration related to data transformation

    def get_data_transformer_object(self): # for creating the preprocessor object which will be used to transform the data
        #this function will create a preprocessor object which will be used to transform the data and it will return the preprocessor object
        try:
            numerical_columns = ['writing_score', 'reading_score'] # list of numerical columns in the dataset
            categorical_columns = ['gender',
                                        'race_ethnicity', 
                                        'parental_level_of_education', 
                                        'lunch', 
                                        'test_preparation_course'] # list of categorical columns in the dataset
                
            num_pipeline = Pipeline(
                steps=[ 
                        ("imputer",SimpleImputer(strategy='median')), # for handling missing values in numerical columns by replacing them with the median value of the column
                        ("scaler",StandardScaler()) # for scaling the numerical columns using standard scaler 
                        
                    ]
                )

            cat_pipeline = Pipeline( # for handling categorical columns by creating a pipeline of transformations which includes handling missing values and encoding the categorical variables
                steps=[
                        ("imputer",SimpleImputer(strategy='most_frequent')), # for handling missing values in categorical columns by replacing them with the most frequent value of the column
                        ("one_hot_encoder",OneHotEncoder()), # for encoding the categorical variables using one hot encoding
                        ("scaler",StandardScaler(with_mean=False)) # for scaling the encoded categorical variables using standard scaler and with_mean=False because we don't want to center the data before scaling
                    ]
                )


            logging.info(f"Categorical columns encoding completed: {categorical_columns}") # logging the categorical columns
            logging.info(f"Numerical columns standard scaling completed: {numerical_columns}") # logging the numerical columns

            preprocessor = ColumnTransformer( # for applying different transformations to different columns using column transformer
                    [
                        ("num_pipeline", num_pipeline, numerical_columns), # for applying the numerical pipeline to the numerical columns
                        ("cat_pipeline", cat_pipeline, categorical_columns) # for applying the categorical pipeline to the categorical columns
                    ]
                )

            return preprocessor # returning the preprocessor object


        except Exception as e:
            logging.info("Error in data transformation") # logging the error in data transformation
            raise CustomException(e, sys) # raising a custom exception with the error message and the system information


    def initiate_data_transformation(self, train_path, test_path): #starting the data transformation process by taking the path of the train and test data as input
        try:
            train_df = pd.read_csv(train_path) # reading the train data from the given path
            test_df = pd.read_csv(test_path) # reading the test data from the given path

            logging.info("Read train and test data completed") # logging the completion of reading train and test data

            preprocessor_obj = self.get_data_transformer_object() # getting the preprocessor object by calling the get_data_transformer_object function

            target_column_name = "math_score" # target column name in the dataset
            numerical_columns = ['writing_score', 'reading_score'] # list of numerical columns in the dataset

            input_feature_train_df = train_df.drop(columns=[target_column_name]) # dropping the target column from the train data to get the input features for training
            target_feature_train_df = train_df[target_column_name] # getting the target column from the train data to get the target feature for training

            input_feature_test_df = test_df.drop(columns=[target_column_name]) # dropping the target column from the test data to get the input features for testing
            target_feature_test_df = test_df[target_column_name] # getting the target column from the test data to get the target feature for testing

            logging.info("Splitting input and target features completed") # logging the completion of splitting input and target features

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) # fitting and transforming the input features of the train data using the preprocessor object
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) # transforming the input features of the test data using the preprocessor object

            logging.info("Applying preprocessing object on training and testing datasets completed") # logging the completion of applying preprocessing object on training and testing datasets

            save_object( # saving the preprocessor object to the given path using the save_object function
                file_path = self.data_transformation_config.preprocessor_obj_file_path, # path to save the preprocessor object
                obj = preprocessor_obj # preprocessor object to be saved
            )

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # concatenating the transformed input features and target feature of the train data to get the final train array
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # concatenating the transformed input features and target feature of the test data to get the final test array

            logging.info("Concatenating transformed input features and target feature completed") # logging the completion of concatenating transformed input features and target feature
        
        except Exception as e:
            logging.info("Error in data transformation") # logging the error in data transformation
            raise CustomException(e, sys) # raising a custom exception with the error message and the system information

                

