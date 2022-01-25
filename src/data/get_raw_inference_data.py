"""
Module to get raw dataset and log it as a versioned artifact.
"""
import wandb

from src.data.get_raw_training_data import get_example_data
from src.utils import log_dataframe
from src.config import config
from src.logger import logger


if __name__ == "__main__":
    with wandb.init(project=config.WANDB_PROJECT, job_type="get-raw-data") as run:
        logger.info("Load raw inference data")
        df = get_example_data().sample(1000)

        logger.info("Creating artifact")
        log_dataframe(
            run=run,
            df=df,
            type="raw-data",
            name="raw-inference-data",
            descr="Raw inference data.",
        )








