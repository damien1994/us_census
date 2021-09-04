"""
Python class for model training
author: Damien Michelle
date: 03/09/2021
"""
import mlflow
from sklearn.ensemble import RandomForestClassifier

from census_predictions.base_logger import logging
from census_predictions.config import USELESS_COLS, LABEL_COL, \
    ENCODING_LABEL, MODEL_OUTPUT_DIR, PARAMS_RANDOMFOREST
from census_predictions.utils import split_data, encode_label, save_model
from census_predictions.MlPipeline import MlPipeline
from census_predictions.EvalCensusModel import EvalCensusModel


class TrainCensusModel(EvalCensusModel, MlPipeline):
    mlflow.set_experiment("Census_predictions_v1")
    #mlflow.sklearn.autolog()

    def __init__(self, train_df, eval_df):
        EvalCensusModel.__init__(self, eval_df)
        self.train_df = train_df

    def compute_census_continuous_training(self):
        #with mlflow.start_run as run:
        ml_pipeline = self.compute_census_model()
        compute_census_eval = EvalCensusModel(self.eval_df)
        compute_census_eval.compute_census_eval(ml_pipeline)

    def compute_census_model(self):
        self.train_df[LABEL_COL] = encode_label(self.train_df, LABEL_COL, ENCODING_LABEL)
        logging.info('SUCCESS - Target encoding has been performed')

        labels, train_data = split_data(self.train_df, LABEL_COL, USELESS_COLS)
        logging.info('SUCCESS - Split between train and labels has been performed')

        model = self.train_randomforest()
        for param in PARAMS_RANDOMFOREST:
            mlflow.log_param(param, model.get_params().get(param))
        ml_pipeline = self.ml_pipeline(train_data, model)
        ml_pipeline.fit(train_data, labels)
        mlflow.sklearn.log_model(ml_pipeline, 'model_pipeline')
        logging.info('SUCCESS - Ml pipeline has been fitted')

        save_model(ml_pipeline, MODEL_OUTPUT_DIR)
        logging.info('SUCCESS - Model has been saved')
        return ml_pipeline

    @staticmethod
    def train_randomforest():
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced_subsample'
        )
