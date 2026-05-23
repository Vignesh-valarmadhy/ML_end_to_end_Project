import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation # for accessing the configuration related to data transformation
from src.components.data_transformation import DataTransformationConfig # for accessing the configuration related to data transformation
from src.components.model_trainer import ModelTrainer # for accessing the configuration related to model training
from src.components.model_trainer import ModelTrainerConfig # for accessing the configuration related to model training

@dataclass # This decorator is used to automatically generate special methods like __init__() and __repr__() for the class.
class DataIngestionConfig: # This class is used to store the configuration for data ingestion. It has three attributes: train_data_path,
    # test_data_path, and raw_data_path.  
    #The default values for these attributes are set to the paths where the train, test, and raw data will be stored.
    train_data_path: str = os.path.join('artifacts', 'train.csv') # This line creates a path for the train data 
    #by joining the 'artifacts' directory with the 'train.csv' file name.
    test_data_path: str = os.path.join('artifacts', 'test.csv') # This line creates a path for the test data
    raw_data_path: str = os.path.join('artifacts', 'data.csv') # This line creates a path for the raw data

class DataIngestion: # This class is responsible for ingesting the data. It has a method called initiate_data_ingestion() #
    #which performs the data ingestion process.
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # This line initializes the ingestion_config 
        #attribute with an instance of the DataIngestionConfig class it will consist of the above 3 line the train, test , raw data path.
    
    def initiate_data_ingestion(self): # This method is responsible for performing the data ingestion process. It reads the raw data, splits it into train and test sets, and saves them to the specified paths.
        logging.info("Entered the data ingestion method or component") # This line logs an informational message indicating that the data ingestion process has started.
        try:
            df = pd.read_csv(os.path.join("notebook", "data", "stud.csv")) # This line reads the raw data from a CSV file and stores it in a DataFrame called df.
            logging.info("Read the dataset as dataframe") # This line logs an informational message indicating that the dataset has been read successfully.

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # This line creates the directory for storing the train data if it does not already exist.

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # This line saves the raw data to a CSV file at the specified path.

            logging.info("Train test split initiated") # This line logs an informational message indicating that the train-test split process has started.

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) # This line splits the DataFrame into a training set and a test set using an 80-20 split.

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # This line saves the training set to a CSV file at the specified path.
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # This line saves the test set to a CSV file at the specified path.

            logging.info("Ingestion of data is completed") # This line logs an informational message indicating that the data ingestion process has been completed.

            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path) # This line returns a tuple containing the paths to the train and test data files.

        except Exception as e: # If any exception occurs during the data ingestion process, it will be caught here.
            raise CustomException(e, sys) # This line raises a custom exception with the original exception and system information.


if __name__ == "__main__":
    obj = DataIngestion() # This line creates an instance of the DataIngestion class and assigns it to the variable obj.
    train_data, test_data = obj.initiate_data_ingestion() # This line calls the initiate_data_ingestion() method on the obj instance, which performs the data ingestion process and returns the paths to the train and test data files. The returned paths are assigned to the variables train_data and test_data, respectively.
    # This block of code checks if the script is being run directly (as the main program) 
    #and if so, it creates an instance of the DataIngestion class and calls the initiate_data_ingestion() 
    # method to start the data ingestion process.

    data_transformation = DataTransformation() # This line creates an instance of the DataTransformation class and assigns it to the
    # variable data_transformation. This instance will be used to perform data transformation on the ingested data.
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data) # This line calls the initiate_data_transformation() method on the data_transformation instance, passing the paths to the train and test data files as arguments. This will start the data transformation process on the ingested data.
    ModelTrainer = ModelTrainer() # This line creates an instance of the ModelTrainer class and assigns
    # it to the variable ModelTrainer. This instance will be used to train the machine learning model on the transformed data.
    print(ModelTrainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)) # This line calls the initiate_model_trainer() method on the ModelTrainer instance, passing the transformed training and test data arrays as arguments. This will start the model training process using the transformed data.
    