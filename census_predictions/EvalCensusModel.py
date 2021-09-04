"""
Python class for model evaluation
author: Damien Michelle
date: 03/09/2021
"""
import pandas as pd
from sklearn.metrics import classification_report

from census_predictions.base_logger import logging
from census_predictions.config import USELESS_COLS, LABEL_COL, ENCODING_LABEL, RESULTS_OUTPUT_DIR, CURRENT_DIR
from census_predictions.utils import split_data, encode_label, load_model, safe_creation_directory


class EvalCensusModel:

    def __init__(self, eval_df):
        self.eval_df = eval_df

    def compute_census_eval(self, ml_pipeline):
        self.eval_df[LABEL_COL] = encode_label(self.eval_df, LABEL_COL, ENCODING_LABEL)
        logging.info('SUCCESS - Target encoding has been performed')

        true_labels, eval_data = split_data(self.eval_df, LABEL_COL, USELESS_COLS)
        logging.info('SUCCESS - Split between train and labels has been performed')

        #ml_pipeline = ml_pipeline if self.retraining else load_model(ml_pipeline)
        predictions = ml_pipeline.predict(eval_data)
            #if self.retraining == 'retraining'\
            #else load_model(ml_pipeline).predict(eval_data)
        logging.info('SUCCESS - Predictions has been computed')

        self.store_classification_report(true_labels, predictions, type(ml_pipeline).__name__, RESULTS_OUTPUT_DIR)

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
