import os
import sys
import time
from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import BaseCrossValidator
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
import src.transform as tr

logger = getLogger(__name__)


class LoadDataset(object):
    def __init__(self,
                 upstream_directory:str,
                 file_prefix: str,
                 file_name: str,
                 ):
        
        self.upstream_directory = upstream_directory
        self.file_prefix = file_prefix
        self.file_name = file_name
        
    def pandas_reader_dataset(self, target:str, time_column:str | None,) -> tuple[pd.DataFrame, pd.Series]:
        file_paths = str(
            Path() / self.upstream_directory / self.file_prefix / self.file_name
        )
        df = pd.read_csv(file_paths)
        if time_column is not None:
            df_ = self.transform_process(df, time_column, target)
            X = df_.drop(labels=[target], axis=1)
            y = df_[target]
            return X, y
        
        X = df.drop(labels=[target], axis=1)
        y = df[target]
        
        return X, y


    def transform_process(self, df:pd.DataFrame, time_column:str="Forecast_time", target:str="load"):
        tr.set_time_index(df,time_column)
        df = tr.create_time_features(df)
        df = tr.create_time_lag_features(df, target=target)
        return df
        

class SolarDataset(object):
    def __init__(self,
                 upstream_directory:str,
                 file_prefix: str,
                 file_name: str,
                 ):
        
        self.upstream_directory = upstream_directory
        self.file_prefix = file_prefix
        self.file_name = file_name
        
    def pandas_reader_dataset(self, target:str, time_column:str | None,) -> tuple[pd.DataFrame, pd.Series]:
        file_paths = str(
            Path() / self.upstream_directory / self.file_prefix / self.file_name
        )
        df = pd.read_csv(file_paths)
        if time_column is not None:
            df_ = self.transform_process(df, time_column, target)
            X = df_.drop(labels=[target], axis=1)
            y = df_[target]
            return X, y
        
        X = df.drop(labels=[target], axis=1)
        y = df[target]
        
        return X, y


    def transform_process(self, df:pd.DataFrame, time_column:str="Forecast_time", target:str="load"):
        tr.set_time_index(df,time_column)
        df = tr.create_time_features(df)
        df = tr.create_time_lag_features(df, target=target)
        hour_group_energy = grouped_frame(df=train, group_col_list=['hour'], target_col_list=['generation'], method='mean')
        hour_group_energy_std = grouped_frame(df=train, group_col_list=['hour'], target_col_list=['generation'], method='std')
        cloud_hour_gruop_energy = grouped_frame(df=train, group_col_list=["cloudy", "hour"], target_col_list=["generation"], method="mean")
        train['hour_mean'] = train.apply(lambda x: hour_group_energy.loc[(hour_group_energy.hour == x['hour']), 'generation_mean'].values[0], axis=1)
        train['hour_std'] = train.apply(lambda x: hour_group_energy_std.loc[(hour_group_energy_std.hour == x['hour']), 'generation_std'].values[0], axis=1)

        train['cloud_hour_std'] = train.apply(lambda x: cloud_hour_gruop_energy.loc[(cloud_hour_gruop_energy.hour == x['hour']) & (
            cloud_hour_gruop_energy.cloudy == x['cloudy']), 'generation_mean'].values[0], axis=1)
        train = transform_cyclic(train, col="hour", max_val=23)
        train = transform_cyclic(train, col="month", max_val=12)
        train = transform_cyclic(train, col="dayofweek", max_val=6)
        train = transform_cyclic(train, col="quarter", max_val=4)
        train = transform_cyclic(train, col="dayofyear", max_val=365)
        train = transform_cyclic(train, col="day", max_val=31)
        return df
        
    
def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    average: str = "macro",
):
    if model_type == "classification":
        metrics = [accuracy_score, precision_score, recall_score, f1_score]
    if model_type == "regression":
        metrics = [
            root_mean_squared_error,
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        ]
    y_pred = model.predict(X_test)
    results = {}
    if model_type == "classification":
        results[model.__class__.__name__] = {
            metric.__name__: (
                metric(y_test, y_pred)
                if metric.__name__ == "accuracy_score"
                else metric(y_test, y_pred, average=average)
            )
            for metric in metrics
        }

    elif model_type == "regression":
        results[model.__name__] = {
            metric.__name__: metric(y_test, y_pred) for metric in metrics
        }

    return pd.DataFrame(results)



def train(
    model: BaseEstimator,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_type: str,
    params: Optional[dict] = None,
):
    if params is not None:
        model.set_params(**params)

    model = Pipeline([("preprocessor", pipe), ("classifier", model)])
    model.fit(X_train, y_train)
    eval_result = evaluate(model, X_valid, y_valid, model_type)

    logger.info(f"model trained")
    return model, eval_result