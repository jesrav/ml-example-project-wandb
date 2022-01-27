"""Module to do batch inference."""
import pickle

import joblib
import pandas as pd
import click
import wandb

from src.config import config
from src.logger import logger
from src.utils import read_dataframe_artifact, log_dataframe


def generate_sample_data(sample_size: int) -> pd.DataFrame:
    return pd.read_parquet(config.MODELLING_DATA_PATH).sample(sample_size)


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
    run = wandb.init(project=config.WANDB_PROJECT, job_type="Batch inference", group=group)

    logger.info("Load model.")
    model_artifact = run.use_artifact("model:prod")
    artifact_path = model_artifact.file()
    model = joblib.load(artifact_path)

    logger.info("Get prediction_data.")
    df = read_dataframe_artifact(run, input_data_artifact)

    logger.info("Predict.")
    df['prediction'] = model.predict(df)
    df['model_version'] = model_artifact.version

    logger.info("Write predictions to db.")
    log_dataframe(
        run=run,
        df=df,
        type="predictions",
        name=output_data_artifact,
        descr="Batch predictions and features.",
    )


if __name__ == '__main__':
    main()






