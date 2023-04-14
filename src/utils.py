import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            #train Model
            model.fit(X_train,y_train)

            #predict Training Data
            y_train_pred = model.predict(X_train)

            #predict Testing data
            y_test_pred = model.predict(X_test)
            #Get R2 Scores for train and test data
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            #getting the results in list
            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        logging.info("Error while predicting and evaluating the model ")
        raise CustomException(e, sys)