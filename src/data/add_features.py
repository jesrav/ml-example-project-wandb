"""
Module to add features.
"""
import hydra
import pandas as pd
import wandb
import click

from src.logger import logger
from src.utils import read_dataframe_artifact, log_dataframe


def add_bedrooms_per_room(df: pd.DataFrame) -> pd.DataFrame:
    "Add average number of bedrooms per room."
    df = df.copy()
    df["avg_bedrooms_per_room"] = df.AveBedrms / df.AveRooms
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_bedrooms_per_room(df)
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
                project=config["main"]["project_name"],
                job_type="add_features",
                group=config["main"]["experiment_name"]
        ) as run:

        clean_data_name = config["artifacts"]["clean_data"]["name"]
        clean_data_version = config["artifacts"]["clean_data"]["version"]
        df = read_dataframe_artifact(run, f"{clean_data_name}:{clean_data_version}")

        logger.info('Add features.')
        df = add_features(df)

        logger.info('Log modelling input data artifact.')
        log_dataframe(
            run=run,
            df=df,
            name=config["artifacts"]["model_input"]["name"],
            type=config["artifacts"]["model_input"]["type"],
            descr=config["artifacts"]["model_input"]["description"],
        )


if __name__ == "__main__":
    main()
