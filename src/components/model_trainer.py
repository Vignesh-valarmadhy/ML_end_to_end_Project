import os 
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
                             )
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object, evaluate_models

@dataclass #decorator for creating data classes
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl') #path to save the trained model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() #initialize the model trainer configuration
    
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], #features for training #the last column is the target variable
                train_array[:,-1], #target variable for training #the last column is the target variable
                test_array[:,:-1], #features for testing #the last column is the target variable
                test_array[:,-1] #target variable for testing #the last column is the target variable
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "K-Neighbors Regressor": KNeighborsRegressor()
            }
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            ## to get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            ## to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            if best_model_score < 0.6: #if the best model score is less than 0.6, then we can say that the model is not good enough
                raise CustomException("No best model found") #raise a custom exception with the message "No best model found"
            

            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}") #log the best model name and the best model score

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, #path to save the trained model
                obj=best_model #the best model object to be saved
            )

            predicted = best_model.predict(X_test) #predict the target variable for the testing data using the best model
            best_model_score = r2_score(y_test, predicted) #calculate the r2 score for the testing data
            return best_model_score #return the best model score

        except Exception as e:
            logging.info("Exception occurred during model training")
            raise CustomException(e, sys)