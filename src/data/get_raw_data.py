"""
Module to get raw dataset and log it as a versioned artifact.
"""
import hydra
import wandb

from src.utils import log_dataframe, get_example_data
from src.logger import logger


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
            project=config["main"]["project_name"],
            job_type="get-raw-data",
            group=config["main"]["experiment_name"]
    ) as run:
        logger.info("Get sample inference data.")
        df = get_example_data(sample_size=config["main"].get("inference_sample_size", None))
        
        logger.info("Creating artifact")
        log_dataframe(run=run, df=df, **config["artifacts"]["raw_data"])


if __name__ == "__main__":
    main()







