"""
Python class where pre-processing pipeline is stored
author: Damien Michelle
date: 03/09/2021
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


class MlPipeline:

    def ml_pipeline(self, df: pd.DataFrame, model):
        """
        Scikit pipeline
        :param df: a pandas dataframe
        :param model: a scikit learn model
        :return: a scikit pipeline
        """
        numerical_features = self.get_numeric_features(df)
        categorical_features = self.get_categorical_features(df)

        transformers = ColumnTransformer([
           ('cat', self.categorical_transformer_pipeline(), categorical_features)
        ])
        return Pipeline([
           ('transformers_pipeline', transformers),
           ('model', model)
        ])

    @staticmethod
    def categorical_transformer_pipeline():
        """
        Categorical transformations stored in
        scikit pipeline
        :return: a scikit pipeline
        """
        return Pipeline([
            ('one_hot_encoding', OneHotEncoder())
        ])

    @staticmethod
    def get_numeric_features(df: pd.DataFrame):
        """
        Get numerical column names
        :param df: a pandas dataframe
        :return: an index of numerical column names
        """
        return df.select_dtypes(exclude=['object']).columns

    @staticmethod
    def get_categorical_features(df: pd.DataFrame):
        """
        Get categorical column names
        :param df: a pandas dataframe
        :return: an index of categorical column names
        """
        return df.select_dtypes(include=['object']).columns
