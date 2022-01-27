"""
Module to do preprocessing of artifacts.
"""
import click
import pandas as pd
import wandb

from src.utils import log_dataframe, read_dataframe_artifact
from src.config import config
from src.logger import logger


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df


@click.command()
@click.option(
    '--input-artifacts-artifact',
    type=str,
)
@click.option(
    '--output-artifacts-artifact',
    type=str,
)
@click.option(
    '--group',
    type=str,
)
def main(input_data_artifact, output_data_artifact, group):
    run = wandb.init(project=config.WANDB_PROJECT, job_type="process_data", group=group)

    df = read_dataframe_artifact(run, input_data_artifact)

    logger.info(f"Preprocess raw artifacts.")
    df = preprocess(df)

    logger.info(f"Log preprocessed artifacts.")
    log_dataframe(
        run=run,
        df=df,
        type="clean-artifacts",
        name=output_data_artifact,
        descr="Cleaned artifacts.",
    )


if __name__ == "__main__":
    main()

