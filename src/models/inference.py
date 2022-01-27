"""Module to do batch inference."""
import hydra
import joblib
import wandb

from src.logger import logger
from src.utils import read_dataframe_artifact, log_dataframe


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    run = wandb.init(
        project=config["main"]["project_name"],
        job_type="batch_inference",
        group=config["main"]["experiment_name"],
    )

    logger.info("Load model.")
    model_name = config['artifacts']['model']['name']
    model_version = config['artifacts']['model']['version']
    try:
        model_artifact = run.use_artifact(f"{model_name}:{model_version}")
    except wandb.errors.CommError as e:
        raise ValueError(f"Trained model version does not exist. From WANDB: {e}")
    artifact_path = model_artifact.file()
    model = joblib.load(artifact_path)

    logger.info("Get model input.")
    model_input_name = config['artifacts']['model_input']['name']
    model_input_version = config['artifacts']['model_input']['version']
    df = read_dataframe_artifact(run, f"{model_input_name}:{model_input_version}")

    logger.info("Predict.")
    df['prediction'] = model.predict(df)
    df['model_version'] = model_artifact.version

    logger.info("Log predictions.")
    log_dataframe(
        run=run,
        df=df,
        type=config['artifacts']['predictions']['type'],
        name=config['artifacts']['predictions']['name'],
        descr="Batch predictions and features.",
    )


if __name__ == '__main__':
    main()






