"""
Module to get raw dataset.
"""
from typing import Optional

import hydra
import wandb
from sklearn.datasets import fetch_california_housing
import pandas as pd

from src.utils import log_dataframe
from src.logger import logger


def get_raw_data(
    sample_size: Optional[int] = None,
    med_inc_mean_drift_percentage: Optional[float] = None, 
    **kwargs
    ) -> pd.DataFrame:
    """Get california housing data."""
    _ = kwargs
    data = fetch_california_housing(as_frame=True)
    df = data.data
    df["median_house_price"] = data.target
    if med_inc_mean_drift_percentage:
        df["MedInc"] = df["MedInc"] * (1 + med_inc_mean_drift_percentage)
    if sample_size:
        return df.sample(sample_size)
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
            project=config["main"]["project_name"],
            job_type="get-raw-data",
            group=config["main"]["experiment_name"]
    ) as run:
        logger.info("Get sample inference data.")
        df = get_raw_data(
            sample_size=config["main"].get("inference_sample_size", None),
            med_inc_mean_drift_percentage=config["main"].get("med_inc_mean_drift_percentage", None)
        )
        
        logger.info("Creating artifact")
        log_dataframe(run=run, df=df, **config["artifacts"]["raw_data"])


if __name__ == "__main__":
    main()







