import logging
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
import wandb

from src.exceptions import ArtifactDoesNoteExistError


logger = logging.getLogger(__name__)


def log_file(run, file_path: str, type: str, name: str, description: Optional[str] = "", **kwargs) -> None:
    _ = kwargs
    artifact = wandb.Artifact(
        type=type,
        description=description,
        name=name,
    )
    artifact.add_file(file_path)

    logger.info(f"Logging artifact file {name}")
    run.log_artifact(artifact)

    artifact.wait()


def log_dir(run, dir_path: str, type: str, name: str, description: Optional[str] = "", **kwargs) -> None:
    _ = kwargs
    artifact = wandb.Artifact(
        type=type,
        description=description,
        name=name,
    )
    artifact.add_dir(dir_path)

    logger.info(f"Logging artifact directory {name}")
    run.log_artifact(artifact)

    artifact.wait()


def log_dataframe(run, df: pd.DataFrame, type: str, name: str, description: Optional[str] = "", **kwargs) -> None:
    _ = kwargs
    with TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "artifacts.parquet"
        df.to_parquet(file_name)
        log_file(run, file_name, type, name, description)


def read_dataframe_artifact(run, name: str, version: str, **kwargs) -> pd.DataFrame:
    artifact_tag = f"{name}:{version}"
    _ = kwargs
    logger.info(f"Downloading artifact {artifact_tag}")
    try:
        artifact = run.use_artifact(artifact_tag)
    except wandb.errors.CommError as e:
        ArtifactDoesNoteExistError(f"Data version does not exist. From WANDB: {e}")
    artifact_path = artifact.file()
    return pd.read_parquet(artifact_path)


def get_model_artifact(project_name: str, model_name: str, model_version: str):
    api = wandb.Api()
    try:
        return api.artifact(f"{project_name}/{model_name}:{model_version}")
    except wandb.errors.CommError as e:
        raise ArtifactDoesNoteExistError(f"Trained model version does not exist. From WANDB: {e}")