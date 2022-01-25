"""Module that contains custom sklearn compatible transformer classes."""
from typing import List

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from pandera.schemas import DataFrameSchema


class DataFrameValidatior(TransformerMixin, BaseEstimator):
    """Transformer that validates the schema of a dataframe

    The validation is done with the pandera package
    """

    def __init__(self, pandera_schema: DataFrameSchema):
        """Constructor method

        :param pandera_schema: Pandera schema
        """
        self.pandera_schema = pandera_schema

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return self.pandera_schema.validate(X)

    def get_params(self, deep=True):
        return {
            "pandera_schema": self.pandera_schema,
        }

    def set_params(self, **kwargs):

        if "pandera_schema" in kwargs:
            setattr(self, "pandera_schema", kwargs["pandera_schema"])


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a subset of columns from a pandas dataframe
    """
    def __init__(self, columns: List[str]):
        """Constructor method

        :param pandera_schema: Pandera schema
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.columns]

    def get_params(self, deep=True):
        return {
            "columns": self.columns,
        }

    def set_params(self, **kwargs):
        if "columns" in kwargs:
            setattr(self, "columns", kwargs["columns"])
