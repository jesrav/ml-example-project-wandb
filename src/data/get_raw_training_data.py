"""
Module to get raw dataset and log it as a versioned artifact.
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
import wandb

from src.utils import log_dataframe
from src.config import config
from src.logger import logger


def get_example_data() -> pd.DataFrame:
    """Get california housing data."""
    data = fetch_california_housing(as_frame=True)
    df = data.data
    df[config.TARGET_COLUMN] = data.target
    return df


if __name__ == "__main__":
    with wandb.init(project=config.WANDB_PROJECT, job_type="get-raw-data") as run:
        logger.info("Load raw data")
        df = get_example_data()

        logger.info("Creating artifact")
        log_dataframe(
            run=run,
            df=df,
            type="raw-data",
            name="raw-data",
            descr="Raw data.",
        )








