"""
Module to get raw dataset and log it as a versioned artifact.
"""
import hydra
import pandas as pd
from sklearn.datasets import fetch_california_housing
import wandb

from src.utils import log_dataframe
from src.logger import logger

TARGET_COLUMN = "median_house_price"


def get_example_data() -> pd.DataFrame:
    """Get california housing artifacts."""
    data = fetch_california_housing(as_frame=True)
    df = data.data
    df[TARGET_COLUMN] = data.target
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
            project=config["main"]["project_name"],
            job_type="get-raw-data",
            group=config["main"]["experiment_name"]
    ) as run:
        logger.info("Get raw training data")
        df = get_example_data()

        logger.info("Creating artifact")
        log_dataframe(
            run=run,
            df=df,
            name=config["artifacts"]["raw_training_data"]["name"],
            type=config["artifacts"]["raw_training_data"]["type"],
            descr=config["artifacts"]["raw_training_data"]["description"],
        )


if __name__ == "__main__":
    main()







