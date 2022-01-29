from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
import wandb
from sklearn.datasets import fetch_california_housing

from src.logger import logger


def get_example_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Get california housing data."""
    data = fetch_california_housing(as_frame=True)
    df = data.data
    df["median_house_price"] = data.target
    if sample_size:
        return df.sample(sample_size)
    return df


def log_file(run, file_path: str, type: str, name: str, descr: Optional[str] = "") -> None:
    artifact = wandb.Artifact(
        type=type,
        description=descr,
        name=name,
    )
    artifact.add_file(file_path)

    logger.info(f"Logging artifact file {name}")
    run.log_artifact(artifact)

    artifact.wait()


def log_dir(run, dir_path: str, type: str, name: str, descr: Optional[str] = "") -> None:
    artifact = wandb.Artifact(
        type=type,
        description=descr,
        name=name,
    )
    artifact.add_dir(dir_path)

    logger.info(f"Logging artifact directory {name}")
    run.log_artifact(artifact)

    artifact.wait()


def log_dataframe(run, df: pd.DataFrame, type: str, name: str, descr: Optional[str] = "") -> None:
    with TemporaryDirectory() as tmpdirname:
        file_name = tmpdirname + "artifacts.parquet"
        df.to_parquet(file_name)
        log_file(run, file_name, type, name, descr)


def read_dataframe_artifact(run, artifact_tag):
    logger.info(f"Downloading artifact {artifact_tag}")
    artifact = run.use_artifact(artifact_tag)
    artifact_path = artifact.file()
    return pd.read_parquet(artifact_path)

