"""
Python class where pre-processing pipeline is stored
author: Damien Michelle
date: 03/09/2021
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, KBinsDiscretizer

from census_predictions.BayesianTargetEncoder import BayesianTargetEncoder


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
            ('num', self.numerical_transformer_pipeline(), numerical_features),
            ('target_encoding', self.target_encoding_pipeline(), categorical_features),
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
    def numerical_transformer_pipeline():
        """
        Categorical transformations stored in
        scikit pipeline
        :return: a scikit pipeline
        """
        return Pipeline([
            ('kbins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans'))
        ])

    @staticmethod
    def target_encoding_pipeline():
        return Pipeline([
            ('bayesian_target_encoding', BayesianTargetEncoder(prior_weight=10, suffix=''))
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
