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

        df = read_dataframe_artifact(run, **config["artifacts"]["raw_data"])

        logger.info('Preprocess raw artifacts.')
        df = preprocess(df)

        logger.info('Log preprocessed artifacts.')
        log_dataframe(run=run, df=df, **config["artifacts"]["clean_data"])


if __name__ == "__main__":
    main()
