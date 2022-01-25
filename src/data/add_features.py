"""
Module to add features.
"""
import pandas as pd
import wandb
import click

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


@click.command()
@click.option(
    '--input-data-artifact',
    type=str,
)
@click.option(
    '--output-data-artifact',
    type=str,
)
@click.option(
    '--group',
    type=str,
)
def main(input_data_artifact, output_data_artifact, group):
    run = wandb.init(project=config.WANDB_PROJECT, job_type="add_features", group=group)

    df = read_dataframe_artifact(run, input_data_artifact)

    logger.info(f"Add features.")
    df = add_features(df)

    log_dataframe(
        run=run,
        df=df,
        type="model-input",
        name=output_data_artifact,
        descr="Data ready for inference.",
    )


if __name__ == "__main__":
    main()
