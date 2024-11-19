
import os
from argparse import ArgumentParser, RawTextHelpFormatter

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor

from src.model import TabularDataset
from src.constants import CV_ENUM
from src.configurations import SolarDataConfigurations, SolarFeatures, LoadDataConfigurations, LoadFeatures, TrainConfigurations


def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    model_type: str,
    cv_type: str,
    n_split: int,
):
    
    if model_type == "load":
        Target = LoadFeatures.TARGET
    elif model_type == "solar":
        Target = SolarFeatures.TARGET
    else:
        raise ValueError("Invalid model type is provided.")
    
    
    train_set = TabularDataset(
        upstream_directory=upstream_directory,
        file_prefix=TrainConfigurations.TRAIN_PREFIX,
        file_name=TrainConfigurations.TRAIN_NAME
    )
    
    test_set = TabularDataset(
        upstream_directory=upstream_directory,
        file_prefix=TrainConfigurations.TEST_PREFIX,
        file_name=TrainConfigurations.TEST_NAME,
    )

    
    X_train, y_train = train_set.pandas_reader_dataset(
        target=Target
    )
    X_test, y_test = test_set.pandas_reader_dataset(
        target=Target
    )
    
    if cv_type == CV_ENUM.simple_cv.value:
        cv = KFold(n_splits=n_split, shuffle=True, random_state=42)
    elif cv_type == CV_ENUM.strat_cv.value:
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    else:
        raise ValueError("Invalid cv type is provided.")
    
    
    
    train(
        model=RandomForestRegressor(),
    )

def main():
    
    parser = ArgumentParser(
        description="Train prediction model",
        formatter_class=RawTextHelpFormatter,
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="load",
        help="load or solar; default load",
    )
    
    parser.add_argument(
        "--upstream",
        type=str,
        default="/opt/data/preprocess",
        help="upstream directory to train the models",
    )
    
    parser.add_argument(
        "--cv_type",
        type=str,
        default=CV_ENUM.simple_cv.value,
        help="General CV method",
    )

    parser.add_argument(
        "--n_split", 
        type=int, 
        default=5, 
        help="CV's n_split",
    )
    

train_dataset = TabularDataset()
