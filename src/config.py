import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class Config:
    SERVER = os.environ["SERVER"]
    DATABASE = os.environ["DATABASE"]

    TRAINING_WANDB_PROJECT = "housing-model-training"
    INFERENCE_WANDB_PROJECT = "housing-model-inference"

    RAW_DATA_PATH = "data/raw/data.parquet"
    PREPROCESSED_DATA_PATH = "data/preprocessed/clean_data.parquet"
    MODELLING_DATA_PATH = "data/preprocessed/modelling_data.parquet"
    TRAIN_DF_PATH = "data/preprocessed/train.parquet"
    VALIDATION_DF_PATH = "data/preprocessed/validation.parquet"
    CROSS_VALIDATION_EVAL_PATH = "model-artifacts/cross_validation"
    HOLD_OUT_EVAL_PATH = "model-artifacts/hold_out"

    REGISTERED_MODEL_NAME = "house-price-model"
    VALIDATION_SET_RATIO = 0.2
    DEFAULT_MODEL_CONFIG = "src.models.model_configs.prod_model_config"
    CROSS_VALIDATION_FOLDS = 5
    TARGET_COLUMN = "median_house_price"


class DevelopmentConfig(Config):
    ENVIRONMENT = "dev"


class TestConfig(Config):
    ENVIRONMENT = "test"


class ProductionConfig(Config):
    ENVIRONMENT = "production"


config_map = {
    "dev": DevelopmentConfig,
    "test": TestConfig,
    "prod": ProductionConfig
}

config = config_map[os.environ["ENVIRONMENT"]]