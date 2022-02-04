"""
Module to add features.
"""
import hydra
import pandas as pd
import wandb

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

        df = read_dataframe_artifact(run, **config["artifacts"]["clean_data"])

        logger.info('Add features.')
        df = add_features(df)

        logger.info('Log modelling input data artifact.')
        log_dataframe(run=run, df=df, **config["artifacts"]["model_input"])


if __name__ == "__main__":
    main()
