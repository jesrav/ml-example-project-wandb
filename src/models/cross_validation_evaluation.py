from tempfile import TemporaryDirectory
from typing import Type

import joblib
from sklearn.model_selection import cross_val_predict
import wandb
import hydra

from src.models.evaluation import RegressionEvaluation
from src.models import models
from src.utils import read_dataframe_artifact, log_dir, log_file
from src.config import config
from src.logger import logger


def train(pipeline_class: Type[models.BasePipeline], pipeline_config: dict):
    run = wandb.init(project=config.WANDB_PROJECT, job_type="Cross validation", )

    wandb.config.update(pipeline_config)

    pipeline = pipeline_class.get_pipeline(**pipeline_config)

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

    logger.info("Logging performance metrics.")
    run.summary.update(model_evaluation.get_metrics())

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

    logger.info("Logging model trained on all data as an artifact.")
    with TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "model.pickle"
        joblib.dump(pipeline, file_name)
        log_file(
           run=run,
           file_path=file_name,
           type="model",
           name="model",
           descr="Trained pipeline"
        )


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    model_class = getattr(models, config["model"]["model_class"])
    train(model_class, config["model"]["params"])


if __name__ == '__main__':
    main()