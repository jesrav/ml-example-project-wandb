"""
Module to do preprocessing of artifacts.
"""
import hydra
import pandas as pd
import wandb

from src.utils import log_dataframe, read_dataframe_artifact
from src.logger import logger


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
            project=config["main"]["project_name"],
            job_type="process-data",
            group=config["main"]["experiment_name"]
    ) as run:

        raw_data_name = config["artifacts"]["raw_data"]["name"]
        raw_data_version = config["artifacts"]["raw_data"]["version"]
        df = read_dataframe_artifact(run, f"{raw_data_name}:{raw_data_version}")

        logger.info(f"Preprocess raw artifacts.")
        df = preprocess(df)

        logger.info(f"Log preprocessed artifacts.")
        log_dataframe(
            run=run,
            df=df,
            name=config["artifacts"]["clean_data"]["name"],
            type=config["artifacts"]["clean_data"]["type"],
            descr=config["artifacts"]["clean_data"]["description"],
        )


if __name__ == "__main__":
    main()

