"""
Module to do preprocessing of data.
"""
import pandas as pd
import wandb

from src.utils import log_dataframe, read_dataframe_artifact
from src.config import config
from src.logger import logger


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df


if __name__ == "__main__":
    run = wandb.init(project=config.TRAINING_WANDB_PROJECT, job_type="process_data")

    df = read_dataframe_artifact(run, "raw-data:latest")

    logger.info(f"Preprocess raw data.")
    df = preprocess(df)

    logger.info(f"Log preprocessed data.")
    log_dataframe(
        run=run,
        df=df,
        type="clean-data",
        name="clean-data",
        descr="Cleaned data.",
    )

