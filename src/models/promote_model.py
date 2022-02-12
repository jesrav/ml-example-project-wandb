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
from src.models.utils import get_model


class SingleModelTest:

    def __init__(self, model, test_data, target_col, max_mae):
        self.model = model
        self.test_data = test_data
        self.target_col = target_col
        self.max_mae = max_mae
        self.model_mae = self._calc_model_mae(model, test_data, target_col)

    @staticmethod
    def _calc_model_mae(model, test_data, target_col):
        predictions = model.predict(test_data)
        evaluation = RegressionEvaluation(
            y_true=test_data[target_col],
            y_pred=predictions
        )
        return evaluation.get_metrics()["mae"]

    def _model_has_ok_mae(self) -> bool:
        return self.model_mae < self.max_mae

    def _model_edge_cases_ok(self) -> bool:
        return True

    @property
    def model_passes_tests(self) -> bool:
        return all([
            self._model_has_ok_mae(), self._model_edge_cases_ok()
        ])

    @property
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
        )
        return f"{mae_message}. {edge_case_message}."


class ChallengerModelTest:
    def __init__(self, model_challenger, model_current, test_data, target_col):
        self.model_challenger = model_challenger
        self.model_current = model_current
        self.test_data = test_data
        self.target_col = target_col
        self.model_challenger_mae = self._calc_model_mae(model_challenger, test_data, target_col)
        self.model_current_mae = self._calc_model_mae(model_current, test_data, target_col)

    @staticmethod
    def _calc_model_mae(model, test_data, target_col):
        predictions = model.predict(test_data)
        evaluation = RegressionEvaluation(
            y_true=test_data[target_col],
            y_pred=predictions
        )
        return evaluation.get_metrics()["mae"]

    @property
    def challenger_model_is_better(self) -> bool:
        return self.model_challenger_mae < self.model_current_mae

    def message(self):
        if self.challenger_model_is_better:
            mae_message = (
                f"Challenger model has MAE of {self.model_challenger_mae}, "
                f"which is better than the current models performance of {self.model_current_mae}"
            )
        else:
            mae_message = (
                f"Challenger model has MAE of {self.model_challenger_mae}, "
                f"which is worse than the current models performance of {self.model_current_mae}"
            )

        return f"{mae_message}."

#
# class PromoteModel:
#
#     def __init__(
#             self,
#             model_challenger,
#             model_current,
#             test_data,
#             single_model_test_class,
#             model_comparisson_class,
#     ):
#         self.model_challenger = model_challenger
#         self.model_current = model_current
#         self.test_data = test_data


# def promote_model(project_name, model_name, model_version):
#     model_artifact = get_model_artifact(project_name, model_name, model_version)
#     model_artifact.aliases.append('prod')
#     model_artifact.save()


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
    loaded_model_challenger = get_model(
        project_name=config["main"]["project_name"],
        model_name=config['artifacts']['model']['name'],
        model_version="latest"
    )
    if "prod" in loaded_model_challenger.wandb_artifact.aliases:
        raise ValueError(
            'Latest trained model is already the production model. Something is wrong.'
        )

    logger.info("Loading current prod model if it exists.")
    try:
        loaded_model_current = get_model(
            project_name=config["main"]["project_name"],
            model_name=config['artifacts']['model']['name'],
            model_version="prod"
        )
    except ArtifactDoesNoteExistError:
        loaded_model_current = None

    logger.info("Running single model tests.")
    run.use_artifact(loaded_model_challenger.wandb_artifact)
    single_model_test = SingleModelTest(
        model=loaded_model_challenger.model,
        test_data=test_data,
        target_col=config["main"]["target_column"],
        max_mae=config["main"]["max_mae_to_promote"]
    )

    if loaded_model_current:
        logger.info("Running challenger tests.")
        run.use_artifact(loaded_model_current.wandb_artifact)
        challenger_model_test = ChallengerModelTest(
            model_challenger=loaded_model_challenger.model,
            model_current=loaded_model_current.model,
            test_data=test_data,
            target_col=config["main"]["target_column"],
        )

    if not single_model_test.model_passes_tests:
        logger.warning(
            f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} not promoted. "
            f"{single_model_test.message}"
        )
        wandb.alert(
            title=f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} not promoted.",
            text=single_model_test.message,
            level=wandb.AlertLevel.WARN,
        )

    elif single_model_test.model_passes_tests and not loaded_model_current:
        logger.info(
            f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} promoted."
            f"{single_model_test.message}"
            f"No model currently in production to compare against."
        )
        wandb.alert(
            title=f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} promoted.",
            text=(
                f"{single_model_test.message}."
                f"No model currently in production to compare against."
            ),
            level=wandb.AlertLevel.INFO,
        )
        loaded_model_challenger.promote_to_prod()

    elif single_model_test.model_passes_tests and challenger_model_test.challenger_model_is_better:
        logger.info(
            f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} promoted."
            f"{single_model_test.message}."
            f"{challenger_model_test.message}."
        )
        wandb.alert(
            title=f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} promoted.",
            text=(
                f"{single_model_test.message}."
                f"{challenger_model_test.message}."
            ),
            level=wandb.AlertLevel.INFO,
        )
        loaded_model_challenger.promote_to_prod()
    else:
        logger.warning(
            f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} not promoted."
            f"{single_model_test.message}."
            f"{challenger_model_test.message}."
        )
        wandb.alert(
            title=f"Trained model with id {loaded_model_challenger.model_meta_data.model_id} not promoted.",
            text=(
                f"{single_model_test.message}."
                f"{challenger_model_test.message}."
            ),
            level=wandb.AlertLevel.WARN,
        )


if __name__ == '__main__':
    main()
