"""
Module for doing drift detection
"""
from curses.panel import version
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import wandb
from evidently.analyzers.data_drift_analyzer import DataDriftAnalyzer
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.model_profile import Profile

from src.logger import logger
from src.utils import read_dataframe_artifact, log_file


def get_model_training_data(run, project_name, model_name, model_version) -> pd.DataFrame:
    """Get training data used to train a specific model"""
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{project_name}/{model_name}:{model_version}")
    except wandb.errors.CommError as e:
        raise ValueError(f"Trained model version does not exist. From WANDB: {e}")
    training_run = artifact.logged_by()
    training_data_artifact_name_and_version =  training_run.used_artifacts()[0]._artifact_name
    training_data_artifact_name, training_data_artifact_version = (
        training_data_artifact_name_and_version.split(":")
    )
    return read_dataframe_artifact(
        run, name=training_data_artifact_name, version=training_data_artifact_version
    )


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    run = wandb.init(
        project=config["main"]["project_name"],
        job_type="drift_detection",
        group=config["main"]["experiment_name"],
    )
    training_data = get_model_training_data(
        run=run,
        project_name=config["main"]["project_name"],
        model_name=config['artifacts']['model']['name'],
        model_version=config['artifacts']['model']['version'],
    )

    # Get data supposed to represent a batch of recent data used for inference.
    # Most likely implemented as a rolling window. In this case we are just getting
    # data from the last batch inference.
    logger.info("Load data used for inference.")
    inference_data = read_dataframe_artifact(run=run, **config['artifacts']['model_input'])

    logger.info("Create and log data drift report.")
    data_drift_report = Dashboard(tabs=[DataDriftTab()])
    data_drift_report.calculate(
        reference_data=training_data,
        current_data=inference_data
    )
    with TemporaryDirectory() as tmpdirname:
        data_drift_report_file_name = tmpdirname + "data_drift_report.html"
        data_drift_report.save(data_drift_report_file_name)
        log_file(
            run=run,
            file_path=data_drift_report_file_name,
            **config["artifacts"]["feature_drift_report"]
        )

    logger.info("Create and log data drift profile.")
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(
        reference_data=training_data,
        current_data=inference_data
    )
    with TemporaryDirectory() as tmpdirname:
        data_drift_profile_file_name = tmpdirname + "data_drift_profile.json"
        with open(data_drift_profile_file_name, "w") as file:
            file.write(data_drift_profile.json())
        log_file(
            run=run,
            file_path=data_drift_profile_file_name,
            **config["artifacts"]["feature_drift_profile"]
        )

    # Get number of drifted features from analyzer
    n_drifted_features = data_drift_profile.analyzers_results[DataDriftAnalyzer].metrics.n_drifted_features

    if n_drifted_features > 0:
        warning_text = (
            f"Feature drift detected for {n_drifted_features} features. "
            f"Check data drift report and profile in run:{run.get_url()}"
        )
        logger.warning(warning_text)
        wandb.alert(
            title='Feature drift detected.',
            text=warning_text,
            level=wandb.AlertLevel.WARN,
        )


if __name__ == '__main__':
    main()


