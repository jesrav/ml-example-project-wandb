"""
Module to add features.
"""
import pandas as pd
import wandb

from src.config import config
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


if __name__ == "__main__":
    run = wandb.init(project=config.TRAINING_WANDB_PROJECT, job_type="add_features")

    df = read_dataframe_artifact(run, "clean-data:latest")

    logger.info(f"Add features.")
    df = add_features(df)

    log_dataframe(
        run=run,
        df=df,
        type="modelling-data",
        name="modelling-data",
        descr="Data ready for modelling.",
    )

