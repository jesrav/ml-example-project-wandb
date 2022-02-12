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
from src.utils import read_dataframe_artifact, get_model_artifact
from src.exceptions import ArtifactDoesNoteExistError


class TestModel:

    def __init__(self, model, test_data, target_col, max_mae):
        self.model = model
        self.test_data = test_data
        self.target_col = target_col
        self.max_mae = max_mae
        self.model_mae = self._calc_model_mae(model, test_data, target_col)

    def _calc_model_mae(self, model, test_data, target_col):
        predictions = model.predict(test_data)
        evaluation = RegressionEvaluation(
            y_true=test_data[target_col],
            y_pred=predictions
        )
        self.model_mae = evaluation.get_metrics()["mae"]

    def _model_has_ok_mae(self) -> bool:
        return self.model_mae < self.max_mae

    def _model_edge_cases_ok(self) -> bool:
        return True

    @property
    def model_passes_tests(self) -> bool:
        return all([
            self._model_has_ok_mae(), self._model_edge_cases_ok()
        ])

    @@property
    def message(self):
        if self._model_has_ok_mae:
            mae_message = (
                f"Model has MAE of {self.model_mae}, which is under the max threshold of {self.max_mae}"
            )
        else:
            mae_message = (
                f"Model has MAE of {self.model_mae}, which is not below the max threshold of {self.max_mae}"
            )
        edge_case_message = (
            "Model passes all edge cases" if self._model_edge_cases_ok else "Model does not pass all edge cases"
        return f"{mae_message}. {edge_case_message}"


class PromoteModel:

    def __init__(
            self,
            model_challenger,
            model_current,
            test_data,
            single_model_test_class,
            model_comparisson_class,
    ):
        self.model_challenger = model_challenger
        self.model_current = model_current
        self.test_data = test_data


def promote_model(project_name, model_name, model_version):
    model_artifact = get_model_artifact(project_name, model_name, model_version)
    model_artifact.aliases.append('prod')
    model_artifact.save()


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
        name=config['artifacts']['test_data']['name'],
        version = "latest"
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

    logger.info("Predicting on hold out test set to get model performance of latest trained model.")
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
    except ArtifactDoesNoteExistError:
        logger.info("No current production model. Promoting model without comparing.")
        promote_model(
            project_name=config["main"]["project_name"],
            model_name=config['artifacts']['model']['name'],
            model_version="latest"
        )
        return

    run.use_artifact(current_prod_model_artifact)
    latest_model_path = current_prod_model_artifact.file()
    current_prod_model = joblib.load(latest_model_path)

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
