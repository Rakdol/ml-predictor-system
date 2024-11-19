import os
from logging import getLogger
from argparse import ArgumentParser, RawTextHelpFormatter

from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor

from src.model import SolarDataset, LoadDataset, get_load_pipeline, get_solar_pipeline, train_cv_models, evaluate
from src.constants import CV_ENUM
from src.configurations import (
    SolarDataConfigurations,
    SolarFeatures,
    LoadDataConfigurations,
    LoadFeatures,
    TrainConfigurations,
)

logger = getLogger(__name__)

def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    model_type: str,
    cv_type: str,
    n_split: int,
):

    if model_type == "load":
        target = LoadFeatures.TARGET
        train_set = LoadDataset(
            upstream_directory=upstream_directory,
            file_prefix=TrainConfigurations.TRAIN_PREFIX,
            file_name=TrainConfigurations.TRAIN_NAME,
        )

        test_set = LoadDataset(
            upstream_directory=upstream_directory,
            file_prefix=TrainConfigurations.TEST_PREFIX,
            file_name=TrainConfigurations.TEST_NAME,
        )
        input_pipeline = get_load_pipeline()
        

    elif model_type == "solar":
        target = SolarFeatures.TARGET

        train_set = SolarDataset(
            upstream_directory=upstream_directory,
            file_prefix=TrainConfigurations.TRAIN_PREFIX,
            file_name=TrainConfigurations.TRAIN_NAME,
        )

        test_set = SolarDataset(
            upstream_directory=upstream_directory,
            file_prefix=TrainConfigurations.TEST_PREFIX,
            file_name=TrainConfigurations.TEST_NAME,
        )
        
        input_pipeline = get_solar_pipeline()

    else:
        raise ValueError("Invalid model type is provided.")

    X_train, y_train = train_set.pandas_reader_dataset(target=target)
    X_test, y_test = test_set.pandas_reader_dataset(target=target)

    if cv_type == CV_ENUM.simple_cv.value:
        cv = KFold(n_splits=n_split, shuffle=True, random_state=42)
    elif cv_type == CV_ENUM.strat_cv.value:
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    elif cv_type == CV_ENUM.time_cv.value:
        cv = TimeSeriesSplit(n_splits=n_split)  
    else:
        raise ValueError("Invalid cv type is provided.")
    
    models = {
                
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "Elastic Regression": ElasticNet(),
                "MLP Regression": MLPRegressor(),
                
            }
    
    
    trained_result, trained_models = train_cv_models(
        models=models,
        pipe=input_pipeline,
        cv=cv,
        X_train=X_train,
        y_train=y_train,
        model_type="regression",
        params=None
    )
    
    print(trained_result)
    # evaluate()


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
        "--downstream",
        type=str,
        default="/opt/artifacts/model",
        help="downstream directory to save the models",
    )

    parser.add_argument(
        "--cv_type",
        type=str,
        default="cv",
        help="General CV method",
    )

    parser.add_argument(
        "--n_split",
        type=int,
        default=5,
        help="CV's n_split",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = args.upstream
    downstream_directory = args.downstream

    if not os.path.exists(downstream_directory):
        os.makedirs(downstream_directory)

    start_run(
        mlflow_experiment_id=mlflow_experiment_id,
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        model_type=args.model_type,
        cv_type=args.cv_type,
        n_split=args.n_split,
    )
