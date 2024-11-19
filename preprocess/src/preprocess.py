import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from logging import getLogger

import mlflow
import pandas as pd
from distutils.dir_util import copy_tree

from src.configurations import (
    SolarDataConfigurations,
    LoadDataConfigurations,
    SolarFeatures,
    LoadFeatures,
)
import src.convert_data as cvd

logger = getLogger(__name__)

def main():
    parser = ArgumentParser(
        description="Preprocess dataset",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="load",
        help="load or solar; default load",
    )

    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/data/preprocess/",
        help="downstream directory",
    )

    parser.add_argument(
        "--cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )

    args = parser.parse_args()

    data_name = args.data.lower()
    downstream_directory = os.path.join(args.downstream, data_name)

    if args.cached_data_id:
        cached_artifact_directory = os.path.join(
            "/tmp/mlruns/",
            args.cached_data_id,
            "artifacts/downstream_directory",
        )
        copy_tree(
            cached_artifact_directory,
            downstream_directory,
        )

    else:

        train_output_destination = os.path.join(
            downstream_directory,
            "train",
        )
        test_output_destination = os.path.join(
            downstream_directory,
            "test",
        )

        if not os.path.exists(downstream_directory):
            os.makedirs(downstream_directory)
        if not os.path.exists(train_output_destination):
            os.makedirs(train_output_destination)
        if not os.path.exists(test_output_destination):
            os.makedirs(test_output_destination)
            
        if data_name == "load":
            logger.info(f"============ data name is {data_name} ===========")
            load = pd.read_csv(LoadDataConfigurations.TARGET_FILE)
            load_ = cvd.convert_load_time(load)
            train, test = cvd.data_spliter(load_)
            header_cols = train.columns.to_list()
            header = ",".join(header_cols)
            cvd.save_to_csv(train, train_output_destination, "train", header)
            cvd.save_to_csv(test, test_output_destination, "test", header)
            
            mlflow.log_artifacts(
                downstream_directory,
                artifact_path="downstream_directory",
            )
            logger.info(f"============ train and test are saved {downstream_directory} ===========")

        elif data_name == "solar":
            logger.info(f"============ data name is {data_name} ===========")
            weather = pd.read_csv(SolarDataConfigurations.WEATHER_FILE)
            location = pd.read_csv(SolarDataConfigurations.LOCATION_FILE)
            energy = pd.read_csv(SolarDataConfigurations.TARGET_FILE)

            cvd.convert_energy_time(energy)
            weather_ = cvd.extract_forecast_weather(weather)
            weather_range = cvd.extract_btw_datetimes(weather_)
            merged_df = cvd.merge_weather_energy(
                weather=weather_range,
                energy=energy,
            )
            train, test = cvd.data_spliter(merged_df)
            header_cols = train.columns.to_list()
            header = ",".join(header_cols)
            location_cols = location.columns.to_list()
            location_header = ",".join(location_cols)
            
            cvd.save_to_csv(train, train_output_destination, "train", header)
            cvd.save_to_csv(test, test_output_destination, "test", header)
            cvd.save_to_csv(
                location, downstream_directory, "location", header=location_header
            )

            mlflow.log_artifacts(
                downstream_directory,
                artifact_path="downstream_directory",
            )
            logger.info(f"============ train and test are saved {downstream_directory} ===========")


if __name__ == "__main__":
    main()
