from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
import wandb

from src.logger import logger


def log_dataframe(run, df: pd.DataFrame, type: str, name: str, descr: Optional[str] = "") -> None:
    with TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "data.parquet"
        df.to_parquet(file_name)

        artifact = wandb.Artifact(
            type=type,
            description=descr,
            name=name,
        )
        artifact.add_file(file_name)

        logger.info(f"Logging artifact {name}")
        run.log_artifact(artifact)

        artifact.wait()


def read_dataframe_artifact(run, arrtifact_tag):
    logger.info(f"Downloading artifact {arrtifact_tag}")
    artifact = run.use_artifact(arrtifact_tag)
    artifact_path = artifact.file()
    return pd.read_parquet(artifact_path)

