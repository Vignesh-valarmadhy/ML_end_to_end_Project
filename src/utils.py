import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException

def save_object(file_path, obj): # for saving the preprocessor object to the given path
    try:
        dir_path = os.path.dirname(file_path) # getting the directory path from the file path
        os.makedirs(dir_path, exist_ok=True) # creating the directory if it doesn't exist
        with open(file_path, 'wb') as file_obj: # opening the file in write binary mode to save the object
            dill.dump(obj, file_obj) # saving the object to the file using dill which is a library for serializing and deserializing Python objects
    except Exception as e:
        raise CustomException(e, sys) # raising a custom exception with the error message and the system information