from typing import Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np


def convert_time(x: str) -> str:
    Ymd, HMS = x.split(" ")
    H, M, S = HMS.split(":")
    H = str(int(H) - 1)
    HMS = ":".join([H, M, S])
    return " ".join([Ymd, HMS])


def to_date(x: int) -> pd.DateOffset:
    return pd.DateOffset(hours=x)


def convert_energy_time(
    energy: pd.DataFrame,
    time_column: str = "time",
) -> None:
    """
    solar energy dataset start time is 01:00 rather than 00:00
    this function convert start time from 01:00 to 00:00
    params: energy: pd.DataFrame, time_column: datetime column name
    return: None
    """

    energy[time_column] = energy[time_column].apply(lambda x: convert_time(x))


def convert_load_time(
    load: pd.DataFrame,
    time_column: str = "timestamp",
    to_column: str = "Forecast_time",
) -> pd.DataFrame:

    load[to_column] = pd.to_datetime(load[time_column])

    return load.drop(labels=[time_column], axis=1)


def extract_forecast_weather(
    fcst_weather: pd.DataFrame,
    time_column: str = "Forecast time",
) -> pd.DataFrame:
    """
    extract forecast weather at 14 o'clock
    The result is typical table data

    Args:
        fcst_weather (pd.DataFrame): forecast weather data provieded by KMA
        time_column (str): Forecast time column name in fcst_weather

    Returns:
        pd.DataFrame: Extracted weather data announced at 14 o'clock
    """

    fcst_weather["Forecast_time"] = pd.to_datetime(fcst_weather[time_column])
    fcst_weather_14 = fcst_weather[fcst_weather["Forecast_time"].dt.hour == 14]
    fcst_weather_14 = fcst_weather_14[
        (fcst_weather_14["forecast"] >= 10) & (fcst_weather_14["forecast"] <= 33)
    ]
    fcst_weather_14["Forecast_time"] = fcst_weather_14[
        "Forecast_time"
    ] + fcst_weather_14["forecast"].map(to_date)
    fcst_weather_14 = fcst_weather_14[
        [
            "Forecast_time",
            "Temperature",
            "Humidity",
            "WindSpeed",
            "WindDirection",
            "Cloud",
        ]
    ]
    return fcst_weather_14


def extract_btw_datetimes(
    fcst_weather: pd.DataFrame,
    time_column: str = "Forecast_time",
    start: str = "2018-03-02 00:00:00",
    end: str = "2021-03-01 23:00:00",
) -> pd.DataFrame:
    """
    extract weather data with specfic date range

    Args:
        fcst_weather (pd.DataFrame): extracted forcast weather at 14 o'clock
        start (str): start datetime string ex) 2018-03-01 00:00:00
        end (str): end datetime string ex) 2021-01-31 23:00:00
        time_column (str): datetime column name ex) Forecast_time

    Returns:
        pd.DataFrame: exracted weather data from start and end
    """
    fcst_ = pd.DataFrame()
    fcst_[time_column] = pd.date_range(start=start, end=end, freq="h")
    fcst_weather[time_column] = pd.to_datetime(fcst_weather[time_column])

    fcst_ = pd.merge(fcst_, fcst_weather, on=time_column, how="outer")

    # The result of outer join has some NaN values,
    # we conduct the linear interpolate method to fill the NaN values
    fcst = fcst_.interpolate()

    return fcst


def merge_weather_energy(
    weather: pd.DataFrame,
    energy: pd.DataFrame,
    time_colunm: str = "Forecast_time",
    start: str = "2018-03-01 00:00:00",
    end: str = "2021-01-31 23:00:00",
) -> pd.DataFrame:
    """
    Merge weather and energy dataset with specific date range

    Args:
        weather (pd.DataFrame): fcst_weather btw time range
        energy (pd.DataFrame): solar energy data set
        time_colunm (str, optional): Defaults to "Forecast_time".
        start (_type_, optional): . Defaults to "2018-03-01 00:00:00".
        end (_type_, optional):  Defaults to "2021-01-31 23:00:00".

    Returns:
        pd.DataFrame: merged dataset weather and energy,
        this will be used to train the machine learning models.
    """

    # we only use ulsan case in this project
    ulsan_energy = pd.DataFrame()
    ulsan_energy[time_colunm] = pd.date_range(start=start, end=end, freq="h")
    ulsan_energy["energy"] = energy["ulsan"]

    df = pd.merge(weather, ulsan_energy, on=time_colunm)

    return df


def data_spliter(
    df: pd.DataFrame, n_rows: int = 744
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test set

    Args:
        df (pd.DataFrame): target df to split into train and test set
        n_rows (int, optional): the number of test set. Defaults to 744 (1 month data).
    """
    train, test = df.iloc[:-n_rows], df.iloc[-n_rows:]

    return train, test


def save_to_csv(
    data: Union[np.array, pd.DataFrame],
    destination: str,
    name_prefix: str,
    header: Optional[str],
) -> None:
    save_dest = Path(destination)
    filename_format = f"{name_prefix}_dataset.csv"
    csv_path = save_dest / filename_format
    df = pd.DataFrame(data, columns=header.split(","))
    df.to_csv(csv_path, index=False)
