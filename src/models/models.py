"""
Module that holds the ml model configs.
Any model config that needs to work with the `src/modelling/train_evaluate.py` module,
must conform to this interface specified in the meta class BaseModelConfig.
Author: Jes Ravnbøl
Created: 2022-01-17
"""
from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pandera as pa
import seaborn as sns
from matplotlib import pyplot as plt

from src.models.custom_transfomer_classes import DataFrameValidatior, ColumnSelector


class BasePipeline(ABC):
    """Base class for ml model config."""

    @staticmethod
    @abstractmethod
    def get_pipeline(**params):
        """Returns SKLearn compatible ml pipeline
        input:
            params: Parameters for sklearn compatible pipeline.
        """
        pass

    @staticmethod
    @abstractmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Saves any plots that are relevant for the fitted pipeline."""
        pass


class RidgePipeline(BasePipeline):
    """Model config for ML pipeline using a logistic regression model."""

    @staticmethod
    def get_pipeline(**params):
        """Get logistic regression pipeline
        The pipeline works on a dataframe and selects the features.
        input:
            params: Parameters for the sklearn compatible pipeline.
        """
        features = [
            'MedInc',
            'HouseAge',
            'AveRooms',
            'AveBedrms',
            'Population',
            'AveOccup',
            'Latitude',
            'Longitude',
            'avg_bedrooms_per_room',
        ]

        schema_prediction_data = pa.DataFrameSchema({
            "MedInc": pa.Column(float, nullable=False, required=True),
            "HouseAge": pa.Column(float, nullable=False, required=True),
            "AveRooms": pa.Column(float, nullable=False, required=True),
            "Population": pa.Column(float, nullable=False, required=True),
            "AveOccup": pa.Column(float, nullable=False, required=True),
            "Latitude": pa.Column(float, nullable=False, required=True),
            "Longitude": pa.Column(float, nullable=False, required=True),
        })

        # Define pipeline
        vanilla_pipeline = Pipeline([
            ("column_selector", ColumnSelector(features)),
            ("column_validator", DataFrameValidatior(pandera_schema=schema_prediction_data)),
            ("regressor", Ridge()
             )
        ])
        return deepcopy(vanilla_pipeline).set_params(**params)

    @staticmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Logreg pipeline does not have any plots for the fitted model."""
        pass


class RandomForestPipeline(BasePipeline):
    """Model config for ML pipeline using a random forest model."""

    @staticmethod
    def get_pipeline(**params):
        """Get random forest pipeline
        The pipeline works on a dataframe and selects the features.
        input:
            params: Parameters for the sklearn compatible pipeline.
        """
        features = [
            'MedInc',
            'HouseAge',
            'AveRooms',
            'AveBedrms',
            'Population',
            'AveOccup',
            'Latitude',
            'Longitude',
            'avg_bedrooms_per_room',
        ]

        schema_prediction_data = pa.DataFrameSchema({
            "MedInc": pa.Column(float, nullable=False, required=True),
            "HouseAge": pa.Column(float, nullable=False, required=True),
            "AveRooms": pa.Column(float, nullable=False, required=True),
            "Population": pa.Column(float, nullable=False, required=True),
            "AveOccup": pa.Column(float, nullable=False, required=True),
            "Latitude": pa.Column(float, nullable=False, required=True),
            "Longitude": pa.Column(float, nullable=False, required=True),
        })

        # Define pipeline
        vanilla_pipeline = Pipeline([
            ("column_selector", ColumnSelector(features)),
            ("column_validator", DataFrameValidatior(pandera_schema=schema_prediction_data)),
            ("regressor", RandomForestRegressor()
             )
        ])
        return deepcopy(vanilla_pipeline).set_params(**params)

    @staticmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Save plot of feature importances for random forest model."""
        rf_features = pipeline["classifier"].feature_names_in_
        rf_feature_importances = pipeline["classifier"].feature_importances_
        feature_importance_df = pd.DataFrame(
            zip(
                rf_features,
                rf_feature_importances,
            ),
            columns=["feature", "importance"],
        ).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(20, 20))
        ax = sns.barplot(x="feature", y="importance", data=feature_importance_df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_xticklabels(), fontsize=24)
        plt.setp(ax.get_yticklabels(), fontsize=24)
        plt.xlabel('feature', fontsize=24)
        plt.ylabel('importance', fontsize=24)
        fig = ax.get_figure()
        fig.subplots_adjust(bottom=0.3)
        fig.savefig(Path(out_dir) / Path("random_forest_feature_importances.png"))
