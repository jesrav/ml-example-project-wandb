import importlib
import pickle
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Type

import click
from sklearn.model_selection import cross_val_predict
import wandb

from src.models.evaluation import RegressionEvaluation
from src.models.model_configs import model_configs
from src.utils import read_dataframe_artifact, log_dir, log_file
from src.config import config
from src.logger import logger


def train(model_config: Type[model_configs.BaseModelConfig]):
    run = wandb.init(project=config.TRAINING_WANDB_PROJECT, job_type="Cross validation")

    pipeline = model_config.get_pipeline()

    logger.info("Read training data.")
    df = read_dataframe_artifact(run, "train-validate-data:latest")

    logger.info("predict on hold out data using cross validation.")
    predictions = cross_val_predict(
        estimator=pipeline,
        X=df,
        y=df[config.TARGET_COLUMN],
        cv=config.CROSS_VALIDATION_FOLDS,
        verbose=3,
    )

    model_evaluation = RegressionEvaluation(
        y_true=df[config.TARGET_COLUMN],
        y_pred=predictions,
    )

    logger.info("train on model on all data")
    pipeline.fit(df, df[config.TARGET_COLUMN])

    logger.info("Logging model evaluation artifacts.")
    with TemporaryDirectory() as tmpdirname:
        model_evaluation.save_evaluation_artifacts(outdir=tmpdirname)
        log_dir(
            run=run,
            dir_path=tmpdirname,
            type="evaluation-artifacts",
            name="evaluation-artifacts",
            descr="Artifacts created when evaluating model performance"
        )

    logger.info("Logging model trained on all data to mlflow.")

    with TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "model.pickle"
        with open(file_name, 'wb') as file:
            pickle.dump(pipeline, file)
            log_file(
                run=run,
                file_path=file_name,
                type="model",
                name="model",
                descr="Trained pipeline"
            )


@click.command()
@click.option(
    '--model-config-class',
    type=str,
    help="Module with model config.",
    default=config.DEFAULT_MODEL_CONFIG
)
def main(model_config_class):
    logger.info(f"Importing model configuration from module {model_config_class}.")
    model_config = getattr(model_configs, model_config_class)
    train(model_config)


if __name__ == '__main__':
    main()