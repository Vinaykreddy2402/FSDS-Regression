import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
# initialise data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv') #creating the data path
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv') 

# create a class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestionconfig = DataIngestionconfig() # assigning the class of DataIngestionconfig to acces the variables from the class

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Starts")
        try:
            df = pd.read_csv(os.path.join('notebooks/data','gems.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestionconfig.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestionconfig.raw_data_path,index = False) #writing the raw data into raw data path

            logging.info('Train Test Split')
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestionconfig.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index = False, header = True)
            logging.info('Ingestion of data is complete')

            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )


        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)


#run data ingestion file
"""
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
"""