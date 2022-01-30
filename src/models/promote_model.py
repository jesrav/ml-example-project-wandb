"""
Script for promoting latest trained model to production if the performance on a hold out set:
- is better than a fixed threshold.
- is better than the current production model.
"""
import hydra
import joblib
import wandb

from src.logger import logger
from src.models.evaluation import RegressionEvaluation
from src.utils import read_dataframe_artifact


class ModelDoesNoteExistError(Exception):
    pass


def get_model_artifact(project_name, model_name, model_version):
    api = wandb.Api()
    try:
        return api.artifact(f"{project_name}/{model_name}:{model_version}")
    except wandb.errors.CommError as e:
        raise ModelDoesNoteExistError(f"Trained model version does not exist. From WANDB: {e}")


def promote_model(project_name, model_name, model_version):
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{project_name}/{model_name}:{model_version}")
    except wandb.errors.CommError as e:
        raise ModelDoesNoteExistError(f"Trained model version does not exist. From WANDB: {e}")

    artifact.aliases.append('prod')
    artifact.save()


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    run = wandb.init(
        project=config["main"]["project_name"],
        job_type="test_and_promote_model",
        group=config["main"]["experiment_name"],
    )

    logger.info("Load hold out test data.")
    test_data = read_dataframe_artifact(
        run=run,
        artifact_tag=f"{config['artifacts']['test_data']['name']}:latest"
    )

    logger.info("Loading latest trained model.")
    latest_model_artifact = get_model_artifact(
        project_name=config["main"]["project_name"],
        model_name=config['artifacts']['model']['name'],
        model_version="latest"
    )
    if "prod" in latest_model_artifact.aliases:
        logger.warning(
            'Latest model is already the production model. No new model to promote.'
        )
        wandb.alert(
            title='Model not promoted.',
            text=f'Latest model is already the production model. No new model to promote',
            level=wandb.AlertLevel.WARN,
        )

        return
    run.use_artifact(latest_model_artifact)
    latest_model_path = latest_model_artifact.file()
    latest_model = joblib.load(latest_model_path)

    logger.info("Predictimng on hold out test set to get model performance of latest trained model.")
    predictions = latest_model.predict(test_data)
    evaluation = RegressionEvaluation(
        y_true=test_data[config["main"]["target_column"]],
        y_pred=predictions
    )
    model_performance = evaluation.get_metrics()["mae"]
    performance_th = config["main"]["max_mae_to_promote"]

    if evaluation.get_metrics()["mae"] > config["main"]["max_mae_to_promote"]:
        warning_text = f"Trained model has MAE of {model_performance}, " \
                       f"which is not below maximum MAE of {performance_th}. Model is not promoted."
        logger.warning(warning_text)
        wandb.alert(
            title='Model not promoted.',
            text=warning_text,
            level=wandb.AlertLevel.WARN,
        )
        return
    else:
        logger.info(
            f"Trained model has MAE of {model_performance}, which is below maximum MAE of {performance_th}."
        )

    logger.info("Loading current prod model.")
    try:
        current_prod_model_artifact = get_model_artifact(
            project_name=config["main"]["project_name"],
            model_name=config['artifacts']['model']['name'],
            model_version="prod"
        )
        run.use_artifact(current_prod_model_artifact)
        current_prod_model_path = current_prod_model_artifact.file()
        current_prod_model = joblib.load(current_prod_model_path)
    except ModelDoesNoteExistError:
        logger.info("No current production model. Promoting model without comparing.")
        promote_model(
            project_name=config["main"]["project_name"],
            model_name=config['artifacts']['model']['name'],
            model_version="latest"
        )
        return

    logger.info("Predicting on hold out test set to get model performance of current prod model.")
    predictions_prod = current_prod_model.predict(test_data)
    evaluation_prod = RegressionEvaluation(
        y_true=test_data[config["main"]["target_column"]],
        y_pred=predictions_prod
    )
    model_performance_prod = evaluation_prod.get_metrics()["mae"]
    if model_performance < model_performance_prod:
        logger.info("Latest model has better performance. Promoting latest model to prod.")
        promote_model(
            project_name=config["main"]["project_name"],
            model_name=config['artifacts']['model']['name'],
            model_version="latest"
        )
        wandb.alert(
            title='Model not promoted.',
            text="Latest model has better performance. The model was promoted to prod.",
            level=wandb.AlertLevel.INFO,
        )
    else:
        warning_text = (
            f"Latest trained model has MAE of {model_performance}, which is not lower than current prod model, "
            f"with a MAE of {model_performance_prod}. "
            f"Model is not promoted."
        )
        logger.warning(warning_text)
        wandb.alert(
            title='New model promoted to prod.',
            text=warning_text,
            level=wandb.AlertLevel.WARN,
        )


if __name__ == '__main__':
    main()
