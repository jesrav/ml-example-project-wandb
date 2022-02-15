"""Module to do batch inference."""
import logging

import hydra
import wandb

from src.utils.artifacts import read_dataframe_artifact, log_dataframe
from src.utils.models import get_model

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    run = wandb.init(
        project=config["main"]["project_name"],
        job_type="batch_inference",
        group=config["main"]["experiment_name"],
    )

    logger.info("Load model.")
    loaded_model = get_model(
        config["main"]["project_name"],
        config['artifacts']['model']["name"],
        config['artifacts']['model']['version']
    )
    run.use_artifact(loaded_model.wandb_artifact)

    logger.info("Get model input.")
    df = read_dataframe_artifact(run, **config['artifacts']['model_input'])

    logger.info("Predict.")
    df['prediction'] = loaded_model.model.predict(df)
    df['model_version'] = loaded_model.model_meta_data.version

    logger.info("Log predictions.")
    log_dataframe(run=run, df=df, **config['artifacts']['predictions'])


if __name__ == '__main__':
    main()






