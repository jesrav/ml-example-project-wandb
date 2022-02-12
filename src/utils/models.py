"""utils for using MLFlow in Azure ML."""
from dataclasses import dataclass

import wandb
import mlflow

from src.exceptions import ArtifactDoesNoteExistError


class MLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for creating a MLFlow pyfunc from a fitted model,
     with a predict method
     """
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


@dataclass
class ModelMetaData:
    """Class for holding metadata on registered models."""
    model_id: str
    version: str
    run_id: str


@dataclass
class LoadedModel:
    """Class for holding both a mlflow pyfunc model and meta data
    on the registered model.
    """
    model: mlflow.pyfunc.PyFuncModel
    model_meta_data: ModelMetaData
    wandb_artifact: wandb.Artifact

    @classmethod
    def from_wandb_artifact(cls, wandb_artifact: wandb.Artifact):
        model_path = wandb_artifact.download()
        model = mlflow.pyfunc.load_model(f'file:{model_path}/model')

        model_meta_data = ModelMetaData(
            model_id=wandb_artifact.id,
            version=wandb_artifact.version,
            run_id=wandb_artifact.logged_by(),
        )
        return LoadedModel(
            model=model, model_meta_data=model_meta_data, wandb_artifact=wandb_artifact
        )

    def promote_to_prod(self):
        self.wandb_artifact.aliases.append('prod')
        self.wandb_artifact.save()


def get_model(project_name: str, model_name: str, model_version: str) -> LoadedModel:
    api = wandb.Api()
    try:
        model_artifact = api.artifact(f"{project_name}/{model_name}:{model_version}")
    except wandb.errors.CommError as e:
        raise ArtifactDoesNoteExistError(f"Trained model version does not exist. From WANDB: {e}")

    return LoadedModel.from_wandb_artifact(model_artifact)

