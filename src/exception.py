import sys
import logging
from src.exception import CustomException


def error_message_detail(error, error_detail: sys):
    _, _,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename 
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format()
    file_name,exc_tb.tb_lineno,str(error) 
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message) # inheriting the properties of the parent class Exception or the init function
        self.error_message = error_message_detail(error_message, error_detail=error_detail) # calling the error_message_detail function to get the error message with details

    def __str__(self):
        return self.error_message
    
    # if __name__ == "__main__":
    #     try:
    #         a = 1 / 0 # this will raise a ZeroDivisionError
    #     except Exception as e:
    #         logging.info("An error occurred: {}".format(e)) # log the error message
    #         raise CustomException(e, sys) # raise the custom exception with the error message and details 
    