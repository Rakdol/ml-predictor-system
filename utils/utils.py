def get_solar_experiment_tags() -> dict[str, str]:

    # Provide an Experiment description that will appear in the UI
    experiment_description = (
        "This is the solar forecasting project. "
        "This experiment contains the produce models for hourly solar energy forecasting."
    )

    # Provide searchable tags that define characteristics of the Runs that
    # will be in this Experiment
    experiment_tags = {
        "project_name": "solar-forecasting",
        "store_dept": "produce",
        "team": "sm",
        "mlflow.note.content": experiment_description,
    }
    return experiment_tags


def get_load_experiment_tags() -> dict[str, str]:
    # Provide an Experiment description that will appear in the UI
    experiment_description = (
        "This is the load forecasting project. "
        "This experiment contains the produce models for hourly electricity load forecasting."
    )

    # Provide searchable tags that define characteristics of the Runs that
    # will be in this Experiment
    experiment_tags = {
        "project_name": "load-forecasting",
        "store_dept": "produce",
        "team": "sm",
        "mlflow.note.content": experiment_description,
    }
    return experiment_tags
