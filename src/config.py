import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class Config:
    SERVER = os.environ["SERVER"]
    DATABASE = os.environ["DATABASE"]

    WANDB_PROJECT = "housing-model"

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