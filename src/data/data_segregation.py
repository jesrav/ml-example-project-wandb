import hydra
import wandb
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.utils import read_dataframe_artifact, log_dataframe


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    with wandb.init(
            project=config["main"]["project_name"],
            job_type="data_segregation",
            group=config["main"]["experiment_name"]
    ) as run:
        model_input_data_name = config["artifacts"]["model_input"]["name"]
        model_input_data_version = config["artifacts"]["model_input"]["version"]
        df = read_dataframe_artifact(run, f"{model_input_data_name}:{model_input_data_version}")

        logger.info(f"Split data in train/validate and test data.")
        train_validate_df, test_df = train_test_split(
            df,
            test_size=config["evaluation"]["test_set_ratio"],
        )

        logger.info(f"Log train/validate and test data artifacts.")
        log_dataframe(
            run=run,
            df=train_validate_df,
            name=config["artifacts"]["train_validate_data"]["name"],
            type=config["artifacts"]["train_validate_data"]["type"],
            descr=config["artifacts"]["train_validate_data"]["description"],
        )
        log_dataframe(
            run=run,
            df=test_df,
            name=config["artifacts"]["test_data"]["name"],
            type=config["artifacts"]["test_data"]["type"],
            descr=config["artifacts"]["test_data"]["description"],
        )


if __name__ == '__main__':
    main()