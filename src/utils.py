import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj): # for saving the preprocessor object to the given path
    try:
        dir_path = os.path.dirname(file_path) # getting the directory path from the file path
        os.makedirs(dir_path, exist_ok=True) # creating the directory if it doesn't exist
        with open(file_path, 'wb') as file_obj: # opening the file in write binary mode to save the object
            dill.dump(obj, file_obj) # saving the object to the file using dill which is a library for serializing and deserializing Python objects
    except Exception as e:
        raise CustomException(e, sys) # raising a custom exception with the error message and the system information
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    try:
        report = {}

        for name, model in models.items():

            param = params[name]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

