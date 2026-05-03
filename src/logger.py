import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log" # log file name with current date and time to avoid overwriting the log file
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE) #cwd is current working directory , logs is the folder name and log_file is the file name
os.makedirs(log_path,exist_ok=True) # creating the logs folder if it does not exist

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE) # log file path

logging.basicConfig(
    filename=LOG_FILE_PATH, # log file path
    format="[%(asctime)s] %(levelname)s - %(message)s", # log message format with time, log level and message
    level=logging.INFO, # log level is set to INFO
)

# if __name__ == "__main__":
#     logging.info("Logging has started.") # log message to indicate that logging has started

