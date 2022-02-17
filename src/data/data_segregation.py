"""
Module to split modelling data into
- One data set for training and validation
- One hold out dataset for the final model performance evaluation
"""
import logging

import hydra
import wandb
from sklearn.model_selection import train_test_split

from src.utils.artifacts import read_dataframe_artifact, log_dataframe
from src.utils.models import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
        project=config["main"]["project_name"],
        job_type="data_segregation",
        group=config["main"]["experiment_name"]
    ) as run:

        logger.info("Fix seed.")
        seed = set_seed()
        run.log({"seed": seed})

        logger.info('Load modelling data.')
        df = read_dataframe_artifact(run, **config["artifacts"]["model_input"])

        logger.info('Split data in train/validate and test data.')
        train_validate_df, test_df = train_test_split(
            df,
            test_size=config["evaluation"]["test_set_ratio"],
        )

        logger.info('Log train/validate and test data.')
        log_dataframe(run=run, df=train_validate_df, **config["artifacts"]["train_validate_data"])
        log_dataframe(run=run, df=test_df, **config["artifacts"]["test_data"])


if __name__ == '__main__':
    main()
