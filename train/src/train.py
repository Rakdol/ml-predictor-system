import os
import json
import joblib
from logging import getLogger
from argparse import ArgumentParser, RawTextHelpFormatter

import mlflow
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

    X_train, y_train = train_set.pandas_reader_dataset(target=target, time_column="Forecast_time")
    
    
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
                "MLP Regression": MLPRegressor(max_iter=2000),
                
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

    mlflow.log_metrics(
        {model_name: model_score for model_name, model_score in trained_result.items()}
    )
    model_train_result_file = os.path.join(downstream_directory, f"model_result_{mlflow_experiment_id}.json")

    with open(model_train_result_file, "w") as f:
        json.dump(trained_result, f)

    mlflow.log_artifact(model_train_result_file)

    best_key = max(trained_result, key=trained_result.get)
    best_model = trained_models[best_key]

    print(f"Best Model with the minimum nmse: {best_key}, {trained_result[best_key]}")
    
    signature = mlflow.models.signature.infer_signature(
        X_train.dropna(),
        best_model.predict(X_train),
    )
    input_sample = X_train.dropna()[:2]

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=input_sample,
    )

    model_file_name = os.path.join(
        downstream_directory,
        f"machine_{model_type}_{mlflow_experiment_id}.joblib",
    )
    
    joblib.dump(best_model, model_file_name)
    mlflow.log_artifact(model_file_name)
    logger.info("Save model in mlflow")

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
    downstream_directory = os.path.join(args.downstream, args.model_type)
    
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


if __name__ == "__main__":
    main()