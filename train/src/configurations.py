from pathlib import Path
import os
from logging import getLogger

from src.constants import PLATFORM_ENUM

PAKAGE_ROOT = Path(__name__).resolve().parents[1]

logger = getLogger(__name__)



class PlatformConfigurations:
    platform = os.getenv("PLATFORM", PLATFORM_ENUM.DOCKER.value)
    if not PLATFORM_ENUM.has_value(platform):
        raise ValueError(
            f"PLATFORM must be one of {[v.value for v in PLATFORM_ENUM.__members__.values()]}"
        )


class SolarDataConfigurations:
    FILE_PATH = os.path.join(PAKAGE_ROOT, "data/raw/solar")
    if PlatformConfigurations.platform == "docker":
        FILE_PATH = "/opt/data/raw/solar"
    TARGET_FILE = os.path.join(FILE_PATH, "energy.csv")
    LOCATION_FILE = os.path.join(FILE_PATH, "site_info.csv")
    WEATHER_FILE = os.path.join(FILE_PATH, "ulsan_fcst_data.csv")


class LoadDataConfigurations:

    FILE_PATH = os.path.join(PAKAGE_ROOT, "data/raw/load")
    if PlatformConfigurations.platform == "docker":
        FILE_PATH = "/opt/data/raw/load"
    TARGET_FILE = os.path.join(FILE_PATH, "loadexample.csv")


class SolarFeatures:
    TARGET = "energy"
    NUM_FEATURES = [
        "Temperature",
        "Humidity",
        "WindSpeed",
        "WindDirection",
    ]
    CAT_FEATURES = [
        "Cloud",
    ]
    DATE_FEATURES = ["Forecast_time"]
    
    SCALE_FEATURES =[
        "Temperature",
        "Humidity",
        "hour_mean",
        "hour_std",
        "cloud_hour_mean"
    ]
    
    

class LoadFeatures:
    TARGET = "Load"
    NUM_FEATURES = [
        "temperature",
        "humidity",
    ]
    CAT_FEATURES = [
        "is_hol",
    ]
    SCALE_FEATURES =[
        "temperature",
        "humidity",
        "hour_mean",
        "hour_std",
        "dayweek_hour_mean"
    ]
    DATE_FEATURES = ["Forecast_time"]


logger.info(f"{SolarDataConfigurations.__name__}: {SolarDataConfigurations.__dict__}")
logger.info(f"{SolarFeatures.__name__}: {SolarFeatures.__dict__}")

logger.info(f"{LoadDataConfigurations.__name__}: {LoadDataConfigurations.__dict__}")
logger.info(f"{LoadFeatures.__name__}: {LoadFeatures.__dict__}")


class TrainConfigurations:
    TRAIN_PREFIX = "train"
    TRAIN_NAME = "train_dataset.csv"
    TEST_PREFIX = "test"
    TEST_NAME = "test_dataset.csv"