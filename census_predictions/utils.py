"""
File to store all functions for census predictions
author: Damien Michelle
date: 03/09/2021
"""
import os
import joblib
import argparse
import pandas as pd

from census_predictions.config import CURRENT_DIR
from census_predictions.base_logger import logging


def create_parser():
    """
    Parser
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data',
        help='csv file to process',
        required=True
    )
    parser.add_argument(
        '--eval_data',
        help='train or eval mode',
        required=True,
        default='train'
    )

    return parser


def parse_args(args):
    """
    Parse arguments
    :param args: raw args
    :return: Parsed arguments
    """
    parser = create_parser()
    return parser.parse_args(args=args)


def read_file(input_path: str, data_config: dict) -> pd.DataFrame:
    """
    Read csv file
    :param input_path: path to csv file
    :param data_config: <name_column> : <type> dictionary of data input
    :return: a pandas dataframe
    """
    return pd.read_csv(input_path, header=None, names=[*data_config.keys()], dtype=data_config)


def split_data(input_dataframe: pd.DataFrame, label_col: list, useless_col=None) -> (pd.Series, pd.DataFrame):
    """
    Split data into train & labels
    :param input_dataframe: a pandas dataframe
    :param label_col: label column
    :param useless_col: cols to drop from model
    :return: a series for label and a pandas dataframe without not useful columns
    """
    cols_to_drop = [*label_col, *useless_col] if useless_col else label_col
    return input_dataframe[label_col[0]], input_dataframe.drop(cols_to_drop, axis=1)


def encode_label(input_dataframe: pd.DataFrame, label_col: list, target_encoding: dict) -> pd.Series:
    """
    Encode numerically label column
    :param input_dataframe: a pandas dataframe
    :param label_col: label column
    :param target_encoding: <value to replace> : <new value> dictionary
    :return: a series with new values encoded
    """
    return input_dataframe[label_col].replace(target_encoding)


def safe_creation_directory(path):
    """
    Check if directory exists, if not, create it
    :param path: path to store eda results
    """
    try:
        full_path = os.path.join(CURRENT_DIR, path)
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
            logging.info(f'SUCCESS - folder has been created at {full_path}')
        else:
            logging.info(f'SUCCESS - output is stored in {full_path}')
    except (OSError, SyntaxError) as err:
        logging.info(f'ERROR - during directory creation: {err}')


def save_model(model, output_dir: str):
    """
    Save the model fitted into a .pkl file
    :param model: model fitted - an scikit learn object in our case
    :param output_dir: path where to store the model
    :return a .pkl into the output path
    """
    try:
        safe_creation_directory(output_dir)
        full_ouput_dir = os.path.join(CURRENT_DIR, output_dir)
        return joblib.dump(model, f'{full_ouput_dir}/{type(model).__name__}.pkl')
    except Exception as err:
        logging.info(f'ERROR - during model dump: {err}')


def load_model(input_path: str):
    """
    Load a .pkl file into a model
    :param input_path: path where the model is stored
    :return a a scikit learn model (in our case - can return other objects)
    """
    try:
        return joblib.load(input_path)
    except (FileNotFoundError, MemoryError) as err:
        logging.info(f'ERROR - during model loading: {err}')
