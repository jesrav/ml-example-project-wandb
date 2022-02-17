"""
Module for training and evaluating a model.

A model configuration that implements the interface found in
src.models.model_pipeliene_configs.BasePipelineConfig is passed supplied through the Hyrda configuration.
"""
from tempfile import TemporaryDirectory
from typing import Type
import logging

import mlflow.pyfunc
from sklearn.model_selection import cross_val_predict
import wandb
import hydra

from src.models.evaluation import RegressionEvaluation
from src.models import model_pipeliene_configs
from src.models.model_pipeliene_configs import BasePipelineConfig
from src.utils.artifacts import read_dataframe_artifact, log_dir, log_file
from src.utils.models import MLFlowModelWrapper

logger = logging.getLogger(__name__)


def train_evaluate(
    pipeline_class: Type[BasePipelineConfig],
    config: dict,
):
    run = wandb.init(
        project=config["main"]["project_name"],
        job_type="cross_validation",
        group=config["main"]["experiment_name"],
        config=dict(config)
    )

    target_column = config["main"]["target_column"]

    logger.info("Load data fro training model.")
    df = read_dataframe_artifact(run, **config["artifacts"]["train_validate_data"])

    logger.info("Initialize ml pipeline object.")
    pipeline = pipeline_class.get_pipeline(**(config["model"]["params"]))

    logger.info("predict on hold out data using cross validation.")
    predictions = cross_val_predict(
        estimator=pipeline,
        X=df,
        y=df[target_column],
        cv=config["evaluation"]["cross_validation_folds"],
        verbose=3,
    )

    model_evaluation = RegressionEvaluation(
        y_true=df[target_column],
        y_pred=predictions,
    )

    logger.info("train on model on all artifacts")
    pipeline.fit(df, df[target_column])

    logger.info("Logging performance metrics.")
    run.summary.update(model_evaluation.get_metrics())

    wandb.log(model_evaluation.get_metrics())

    logger.info("Logging model evaluation artifacts.")
    with TemporaryDirectory() as tmpdirname:
        model_evaluation.save_evaluation_artifacts(out_dir=tmpdirname)
        pipeline_class.save_fitted_pipeline_plots(pipeline, out_dir=tmpdirname)
        log_dir(run=run, dir_path=tmpdirname, **config["artifacts"]["evaluation"])

    logger.info("Logging model trained on all data as an artifact.")
    with TemporaryDirectory() as tmpdirname:
        mlflow.pyfunc.save_model(
            python_model=MLFlowModelWrapper(pipeline),
            path=f'{tmpdirname}/model',
            conda_env=pipeline_class.get_conda_env(),
            code_path=["src"],
        )

        log_dir(run=run, dir_path=tmpdirname, **config["artifacts"]["model"])


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    model_class = getattr(model_pipeliene_configs, config["model"]["ml_pipeline_config"])
    train_evaluate(
        pipeline_class=model_class,
        config=config,
    )


if __name__ == '__main__':
    main()