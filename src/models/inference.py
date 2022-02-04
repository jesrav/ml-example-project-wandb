"""Module to do batch inference."""
import hydra
import joblib
import wandb

from src.logger import logger
from src.utils import read_dataframe_artifact, log_dataframe, get_model_artifact


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    run = wandb.init(
        project=config["main"]["project_name"],
        job_type="batch_inference",
        group=config["main"]["experiment_name"],
    )

    logger.info("Load model.")
    model_artifact = get_model_artifact(
        config["main"]["project_name"], 
        config['artifacts']['model']["name"],
        config['artifacts']['model']['version']
    )
    run.use_artifact(model_artifact)
    artifact_path = model_artifact.file()
    model = joblib.load(artifact_path)

    logger.info("Get model input.")
    df = read_dataframe_artifact(run, **config['artifacts']['model_input'])

    logger.info("Predict.")
    df['prediction'] = model.predict(df)
    df['model_version'] = model_artifact.version

    logger.info("Log predictions.")
    log_dataframe(run=run, df=df, **config['artifacts']['predictions'])


if __name__ == '__main__':
    main()






