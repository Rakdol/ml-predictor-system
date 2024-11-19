import os
import sys
import time
import joblib
from logging import getLogger
from argparse import ArgumentParser, RawTextHelpFormatter

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    root_mean_squared_error
)

from src.configurations import SolarDataConfigurations, SolarFeatures, LoadFeatures, LoadDataConfigurations, TrainConfigurations

logger = getLogger(__name__)

class Regressor(object):

    def __init__(
            self,
            model_directory:str,
            file_name:str,
            ):
        
        self.model = self.get_model(model_directory, file_name)

    def get_model(self, model_directory:str, file_name:str):
        model_file_direcotry = os.path.join(model_directory, file_name)
        model = joblib.load(model_file_direcotry)
        return model
    
    def predict(self, x):
        pred = self.mode.predict(x)
        return pred
    

def evaluate(
        test_data_directory:str,
        model_directory:str,
        model_file_name:str,
    ):

    regressor = Regressor(model_directory=model_directory, file_name=model_file_name)


    test_set = LoadDataset(
            upstream_directory=upstream_directory,
            file_prefix=TrainConfigurations.TEST_PREFIX,
            file_name=TrainConfigurations.TEST_NAME,
        )