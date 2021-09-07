"""
Python class for model training
author: Damien Michelle
date: 03/09/2021
"""
import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import classification_report, precision_score, \
    recall_score, f1_score, roc_auc_score

from census_predictions.base_logger import logging
from census_predictions.config import USELESS_COLS, LABEL_COL, ENCODING_LABEL, \
    CURRENT_DIR, MODEL_OUTPUT_DIR, RESULTS_OUTPUT_DIR, space_rf_hyperparameters
from census_predictions.utils import split_data, encode_label, save_model, \
    safe_creation_directory
from census_predictions.MlPipeline import MlPipeline


class CensusModel(MlPipeline):
    mlflow.set_experiment("Census_predictions_v1")
    mlflow_cb = MLflowCallback(
        tracking_uri='mlruns'
    )
    train_labels = None
    train_data = None
    true_labels = None
    eval_data = None

    def __init__(self, train_df, eval_df):
        self.train_df = train_df
        self.eval_df = eval_df

    def compute_census_continuous_training(self):
        """
        Compute ml pipeline for census data
        :return:
        """
        self.train_labels, self.train_data = \
            self.preprocess_data(self.train_df, LABEL_COL, ENCODING_LABEL, USELESS_COLS)
        self.true_labels, self.eval_data = \
            self.preprocess_data(self.train_df, LABEL_COL, ENCODING_LABEL, USELESS_COLS)
        self.study_best_model()

    @staticmethod
    def preprocess_data(df, label_col, target_encoding, useless_col=None):
        """
        TO DO
        :param df:
        :param label_col:
        :param target_encoding:
        :param useless_col:
        :return:
        """
        df[label_col] = encode_label(df, label_col, target_encoding)
        return split_data(df, label_col, useless_col)

    def study_best_model(self):
        """
        Run study from optuna
        :return:
        """
        study = optuna.create_study(study_name='census_predictions',
                                    direction='maximize',
                                    pruner=optuna.pruners.HyperbandPruner(max_resource="auto"))
        study.optimize(self.objective, n_trials=30, callbacks=[self.mlflow_cb])
        return study.best_trial.params, study.best_value

    def objective(self, trial):
        with mlflow.start_run(nested=True):
            hp_space = space_rf_hyperparameters(trial)
            class_weight = self.compute_custom_weights(self.train_labels)
            hp_space['class_weight'] = class_weight
            model = self.train_randomforest(**hp_space)
            for param in [*hp_space.keys()]:
                mlflow.log_param(param, model.get_params().get(param))
            ml_pipeline = self.compute_model(model, self.train_labels, self.train_data)
            metrics = self.eval_model_performance(ml_pipeline, self.true_labels, self.eval_data)
            return metrics.get('f1_score')

    def compute_model(self, model, train_labels, train_data):
        ml_pipeline = self.ml_pipeline(train_data, model)
        ml_pipeline.fit(train_data, train_labels)
        mlflow.sklearn.log_model(ml_pipeline.named_steps['model'], 'model_pipeline')
        return ml_pipeline

    def eval_model_performance(self, ml_pipeline, true_labels: pd.Series, eval_data: pd.DataFrame) -> dict:
        """
        Eval model performance
        :param ml_pipeline: ml scikit pipeline
        :param true_labels: true labels
        :param eval_data: eval data
        :return: a dict with precision, recall, f1_score and roc_auc
        """
        predictions = ml_pipeline.predict(eval_data)
        metrics = self.compute_metrics_score(true_labels, predictions)
        for metric in [*metrics.keys()]:
            mlflow.log_metric(metric, metrics[metric])
        self.store_classification_report(true_labels,
                                         predictions,
                                         type(ml_pipeline.named_steps['model']).__name__,
                                         RESULTS_OUTPUT_DIR)
        return metrics

    @staticmethod
    def train_randomforest(**kwargs):
        return RandomForestClassifier(
            **kwargs,
            n_jobs=-1,
            random_state=42,
            criterion='entropy'
        )

    @staticmethod
    def compute_custom_weights(label_series: pd.Series) -> dict:
        """
        Compute custom weights for imbalanced data
        :param label_series: label col values
        :return: a dict with weights for 0 and 1 custom weights
        """
        total = label_series.shape[0]
        neg = label_series[label_series == 0].shape[0]
        pos = label_series[label_series == 1].shape[0]

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        del total, neg, pos

        return {
            0: weight_for_0,
            1: weight_for_1
        }

    @staticmethod
    def store_classification_report(true_labels, predictions, model_name, output_dir):
        """
        Compute classification report thanks to scikit learn function.
        Returns some metrics like precision, recall, accuracy, ...
        and save it into a csv file
        :param true_labels: true values to compare with predictions
        :param predictions: predictions made with the model trained
        :param model_name: name of the model used like 'LogisticRegression' or 'RandomForest'
        :param output_dir: directory where to store result
        :returns: a csv file where report result is stored
        """
        try:
            safe_creation_directory(RESULTS_OUTPUT_DIR)
            report = classification_report(true_labels, predictions, output_dict=True)
            return pd.DataFrame(report).transpose().to_csv(f'{CURRENT_DIR}/{output_dir}'
                                                           f'/{model_name}_classification_report.csv')
        except (ValueError, TypeError, FileNotFoundError) as err:
            logging.info(f'ERROR - during classification report compute : {err}')

    @staticmethod
    def compute_metrics_score(true_labels, predictions) -> dict:
        """
        Compute precision, recall, f1 score and auc_score metrics
        :param true_labels: true values to compare with predictions
        :param predictions: predictions made with the model trained
        :return: a dict which contains metrics performance
        """
        return {'recall': round(recall_score(true_labels, predictions), 4),
                'precision': round(precision_score(true_labels, predictions), 4),
                'roc_auc': round(roc_auc_score(true_labels, predictions), 4),
                'f1_score': round(f1_score(true_labels, predictions), 4)}