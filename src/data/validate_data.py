"""
Module to validate model input data.
"""
import logging

import hydra
import pandas as pd
import pandera as pa
import wandb

from src.utils.artifacts import read_dataframe_artifact

logger = logging.getLogger(__name__)


def validate_model_input(df: pd.DataFrame) -> pd.DataFrame:
    schema_model_input = pa.DataFrameSchema({
        "MedInc": pa.Column(float, nullable=False, required=True),
        "HouseAge": pa.Column(float, nullable=False, required=True),
        "AveRooms": pa.Column(float, nullable=False, required=True),
        "Population": pa.Column(float, nullable=False, required=True),
        "AveOccup": pa.Column(float, nullable=False, required=True),
        "Latitude": pa.Column(float, nullable=False, required=True),
        "Longitude": pa.Column(float, nullable=False, required=True),
    })
    return schema_model_input.validate(df)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
        project=config["main"]["project_name"],
        job_type="validate-data",
        group=config["main"]["experiment_name"]
    ) as run:

        logger.info('Read model input data.')
        df = read_dataframe_artifact(run, **config["artifacts"]["model_input"])

        logger.info('Validate model input.')
        df = validate_model_input(df)


if __name__ == "__main__":
    main()




