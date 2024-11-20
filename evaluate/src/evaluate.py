import os
import sys
import time
import joblib
import json
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
    root_mean_squared_error,
)

from src.configurations import TrainConfigurations
from src.model import SolarDataset, SolarFeatures, LoadDataset, LoadFeatures
from src.metrics import NMAE, SMAPE, MAPE
from src.utils import plot_evaluated_model_result, plot_metric_results

logger = getLogger(__name__)


class Regressor(object):

    def __init__(
        self,
        model_directory: str,
        file_name: str,
    ):

        self.model = self.get_model(model_directory, file_name)

    def get_model(self, model_directory: str, file_name: str):
        model_file_direcotry = os.path.join(model_directory, file_name)
        model = joblib.load(model_file_direcotry)
        return model

    def predict(self, x):
        pred = self.model.predict(x)
        return pred


def evaluate(
    mlflow_experiment_id: int,
    model_type: str,
    test_data_directory: str,
    model_directory: str,
):

    if model_type == "load":
        target = LoadFeatures.TARGET

        test_set = LoadDataset(
            upstream_directory=test_data_directory,
            file_prefix=TrainConfigurations.TEST_PREFIX,
            file_name=TrainConfigurations.TEST_NAME,
        )

        model_file_name = f"machine_{model_type}_{mlflow_experiment_id}.joblib"
        regressor = Regressor(
            model_directory=model_directory, file_name=model_file_name
        )
        X_test, y_test = test_set.pandas_reader_dataset(
            target=target, time_column="Forecast_time"
        )

        y_pred = regressor.predict(X_test)

        evaluation = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "mape": MAPE(y_test, y_pred),
            "smape": SMAPE(y_test, y_pred),
            "accurcy": 100 - SMAPE(y_test, y_pred),
        }

        result_fig = plot_evaluated_model_result(y_test, y_pred)
        eval_fig = plot_metric_results(evaluation)

        mlflow.log_figure(result_fig, "pred_result_figure.png")
        mlflow.log_figure(eval_fig, "eval_result_figure.png")

        return evaluation

    elif model_type == "solar":
        target = SolarFeatures.TARGET

        test_set = SolarDataset(
            upstream_directory=test_data_directory,
            file_prefix=TrainConfigurations.TEST_PREFIX,
            file_name=TrainConfigurations.TEST_NAME,
        )

        model_file_name = f"machine_{model_type}_{mlflow_experiment_id}.joblib"
        regressor = Regressor(
            model_directory=model_directory, file_name=model_file_name
        )
        X_test, y_test = test_set.pandas_reader_dataset(
            target=target, time_column="Forecast_time"
        )

        y_pred = regressor.predict(X_test)

        evaluation = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "nmae": NMAE(y_test, y_pred, 500),
            "accurcy": 100 - NMAE(y_test, y_pred, 500),
        }

        result_fig = plot_evaluated_model_result(y_test, y_pred)
        eval_fig = plot_metric_results(evaluation)

        mlflow.log_figure(result_fig, "pred_result_figure.png")
        mlflow.log_figure(eval_fig, "eval_result_figure.png")

        return evaluation

    else:
        raise ValueError("Invalid model type, should provide load or solar")


def main():
    parser = ArgumentParser(
        description="evaluate predictor model",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="load", help="prediction model type load or solar"
    )

    parser.add_argument(
        "--upstream",
        type=str,
        default="/opt/artifacts/model",
        help="upstream directory",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/artifacts/evaluate/",
        help="downstream diretory",
    )
    parser.add_argument(
        "--test_parent_directory",
        type=str,
        default="/opt/data/preprocess/",
        help="test data directory",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    model_type = args.model
    upstream_directory = args.upstream
    downstream_directory = args.downstream
    test_parent_directory = args.test_parent_directory

    if not os.path.exists(downstream_directory):
        os.makedirs(downstream_directory)

    logger.info(f".... start {model_type} test evaluate ....")
    evaluations = evaluate(
        mlflow_experiment_id=mlflow_experiment_id,
        model_type=model_type,
        test_data_directory=test_parent_directory,
        model_directory=upstream_directory,
    )
    logger.info(f".... end {model_type} test evaluate ....")

    for k, eval in evaluations.items():
        mlflow.log_metric(k, round(eval, 3))

    log_file = os.path.join(
        downstream_directory, f"mlflow_experiment_id_{mlflow_experiment_id}.json"
    )

    with open(log_file, "w") as f:
        json.dump(evaluations, f)

    mlflow.log_artifact(log_file)


if __name__ == "__main__":
    main()
