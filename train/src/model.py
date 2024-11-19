import os
import sys
import time
from logging import getLogger
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import BaseCrossValidator, cross_val_score
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
from src.configurations import LoadFeatures, SolarFeatures

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
        hour_group_energy = tr.grouped_frame(df=df, group_col_list=['hour'], target_col_list=[target], method='mean')
        hour_group_energy_std = tr.grouped_frame(df=df, group_col_list=['hour'], target_col_list=[target], method='std')
        dayweek_hour_gruop_energy = tr.grouped_frame(df=df, group_col_list=["dayofweek", "hour"], target_col_list=[target], method="mean")
        df['hour_mean'] = df.apply(lambda x: hour_group_energy.loc[(hour_group_energy.hour == x['hour']), f'{target}_mean'].values[0], axis=1)
        df['hour_std'] = df.apply(lambda x: hour_group_energy_std.loc[(hour_group_energy_std.hour == x['hour']), f'{target}_std'].values[0], axis=1)
        df['dayweek_hour_mean'] = df.apply(lambda x: dayweek_hour_gruop_energy.loc[(dayweek_hour_gruop_energy.hour == x['hour']) & (
            dayweek_hour_gruop_energy.cloudy == x['cloudy']), f'{target}_mean'].values[0], axis=1)
        
        return df
        

def get_load_pipeline():
    
    numeric_features = LoadFeatures.SCALE_FEATURES
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )

    categorical_features = LoadFeatures.CAT_FEATURES
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


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
        
        hour_group_energy = tr.grouped_frame(df=df, group_col_list=['hour'], target_col_list=[target], method='mean')
        hour_group_energy_std = tr.grouped_frame(df=df, group_col_list=['hour'], target_col_list=[target], method='std')
        cloud_hour_gruop_energy = tr.grouped_frame(df=df, group_col_list=["cloudy", "hour"], target_col_list=[target], method="mean")
        df['hour_mean'] = df.apply(lambda x: hour_group_energy.loc[(hour_group_energy.hour == x['hour']), f'{target}_mean'].values[0], axis=1)
        df['hour_std'] = df.apply(lambda x: hour_group_energy_std.loc[(hour_group_energy_std.hour == x['hour']), f'{target}_std'].values[0], axis=1)

        df['cloud_hour_mean'] = df.apply(lambda x: cloud_hour_gruop_energy.loc[(cloud_hour_gruop_energy.hour == x['hour']) & (
            cloud_hour_gruop_energy.cloudy == x['cloudy']), f'{target}_mean'].values[0], axis=1)
        
        df = tr.transform_cyclic(df, col="hour", max_val=23)
        df = tr.transform_cyclic(df, col="month", max_val=12)
        df = tr.transform_cyclic(df, col="dayofweek", max_val=6)
        df = tr.transform_cyclic(df, col="quarter", max_val=4)
        df = tr.transform_cyclic(df, col="dayofyear", max_val=365)
        df = tr.transform_cyclic(df, col="day", max_val=31)
        df = tr.convert_wind(df=df, speed="WindSpeed", direction="WindDirection")
        df = tr.convert_cloudy(df=df, column="Cloud", Forecast=True)
        
        return df

def get_solar_pipeline():
    
    numeric_features = SolarFeatures.SCALE_FEATURES
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )

    categorical_features = SolarFeatures.CAT_FEATURES
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor 
    
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


def train_cv_models(
    models: Dict[str, BaseEstimator],  # Fixed the type annotation for dictionary
    pipe: Pipeline,
    cv: BaseCrossValidator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    params: Optional[Dict[str, dict]] = None,  # Fixed type annotation for nested dictionary
):
    # Validate model type and set appropriate scoring metric
    if model_type == "regression":
        indicators = 'neg_mean_squared_error'
    elif model_type == "classification":
        indicators = 'f1_macro'
    else:
        raise ValueError("Invalid model type provided. Must be 'regression' or 'classification'.")

    trained_result = {}
    trained_models = {}

    for key, model in models.items():  
        model_ = Pipeline([("preprocessor", pipe), ("model", model)])  # Renamed "regressor" to "model" for generalization
        logger.info(f"Training model: {key}")

        # Perform cross-validation
        scores = cross_val_score(model_, X_train, y_train, cv=cv, scoring=indicators)
        trained_result[key] = np.mean(scores)

        # Train the model on the full training data
        trained_models[key] = model_.fit(X_train, y_train)

    logger.info("All models trained successfully.")
    return trained_result, trained_models


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
