import os
import sys
from datetime import datetime
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
    
    parser.add_argument(
        "--train_model_type",
        type=str,
        default="load",
        help="forecasting model type load or solar"
    )
    
    parser.add_argument(
        "--train_upstream",
        type=str,
        default="/opt/data/preprocess"
    )
    
    parser.add_argument(
        "--train_downstream",
        type=str,
        default="/opt/model",
        help="downstream directory",
    )
    
    parser.add_argument(
        "--train_cv_type",
        type=str,
        default="cv",
        help="general cv method",
    )
    
    parser.add_argument(
        "--train_n_split",
        type=int,
        default=5,
        help="CV's n_split",
    )
    
    parser.add_argument(
        "--evaluate_model",
        type=str,
        default="load"
    )
    
    parser.add_argument(
        "--evaluate_upstream",
        type=str,
        default="/opt/artifacts/model/",
        help="evaluate upstream directory",
    )
    
    parser.add_argument(
        "--evaluate_downstream",
        type=str,
        default="/opt/artifacts/evaluate/",
        help="evaluate downstream directory",
    )
    
    parser.add_argument(
        "--evaluate_test_parent_directory",
        type=str,
        default="/opt/data/preprocess",
        help="evaluate_test_parent_directory",
    )


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
        
        train_upstream = os.path.join(
            "/mlflow/tmp/mlruns/",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )
        
        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            parameters={
                "upstream": train_upstream,
                "downstream": args.train_downstream,
                "model_type": args.train_model_type,
                "cv_type": args.train_cv_type,
                "n_split": args.train_n_split,
            }
        )
        
        logger.info(
            f"""
                     Train ML project has been completed, 
                     upstream: {train_upstream},
                     downstream: {args.train_downstream},
                     model_type: {args.train_model_type},
                     cv_type: {args.train_cv_type},
                     n_split: {args.train_n_split},
                     """
        )
        
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)
        
        evaluate_upstream = os.path.join(
            "/mlflow/tmp/mlruns/",
            str(mlflow_experiment_id),
            train_run.info.run_id,
            "artifacts/model/",
        )

        logger.info(f".... Evaluate MLproject start ....")
        evaluate_run = mlflow.run(
            uri="./evaluate",
            entry_point="evaluate",
            backend="local",
            parameters={
                "model": data_name,
                "upstream": evaluate_upstream,
                "downstream": args.evaluate_downstream,
                "test_parent_directory": train_upstream,
            },
        )

        logger.info(
            f"""
                     Evaluate ML project has been completed, 
                     model: {data_name}
                     upstream: {evaluate_upstream},
                     downstream: {args.evaluate_downstream},
                     test_parent_directory: {train_upstream}
                     """
        )
        evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run.run_id)

        current_date = datetime.now().strftime("%Y-%m-%d")
        mlflow.set_tag("Release Date", value=current_date)
        mlflow.set_tag("Release Model", value="RandomForest")

if __name__ == "__main__":
    main()
