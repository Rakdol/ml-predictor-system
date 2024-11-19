import os
import sys
from logging import getLogger
from argparse import ArgumentParser, RawTextHelpFormatter

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pprint import pprint

from utils.utils import get_solar_experiment_tags, get_load_experiment_tags

logger = getLogger(__name__)


def get_or_create_experiment(name, tags):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(name=name, tags=tags)

def main():

    parser = ArgumentParser(
        description="Main Pipeline", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--preprocess_data",
        type=str,
        default="load",
        help="load or solar; default load",
    )

    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="/opt/data/preprocess",
        help="preprocess downstream directory",
    )

    parser.add_argument(
        "--preprocess_cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )

    client = MlflowClient()

    args = parser.parse_args()
    data_name = args.preprocess_data
    if data_name == "load":
        experiment_tags = get_load_experiment_tags()
    elif data_name == "solar":
        experiment_tags = get_solar_experiment_tags()

    experiment_name = f"{data_name}_models"
    
    prediction_experiment = get_or_create_experiment(
            name=experiment_name, tags=experiment_tags
        )
        # print(prediction_experiment)

    # mlflow.set_tracking_uri("http://127.0.0.1:8080")
    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))
    # run_name = f"run_{experiment_name}_"
    with mlflow.start_run() as run:
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            backend="local",
            parameters={
                "data": args.preprocess_data,
                "downstream": args.preprocess_downstream,
                "cached_data_id": args.preprocess_cached_data_id,
            },
        )

        logger.info(
            f"""
                     Preprocess ML project has been completed, 
                     data: {args.preprocess_data},
                     downstream: {args.preprocess_downstream},
                     cached_data_id: {args.preprocess_cached_data_id},
                     """
        )

        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)
        
        dataset = os.path.join(
            "/tmp/mlruns/",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )

if __name__ == "__main__":
    main()
