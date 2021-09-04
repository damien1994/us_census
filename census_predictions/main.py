"""
Main file where the census modelisation block
is computed
author: Damien Michelle
date: 04/09/2021
"""
import sys

from census_predictions.utils import parse_args, read_file
from census_predictions.config import COLS
from census_predictions.TrainCensusModel import TrainCensusModel
from census_predictions.EvalCensusModel import EvalCensusModel


def main(train_df, eval_df):
    """
    Main function where Census data intelligence is computed
    """
    census_continous_training = TrainCensusModel(train_df, eval_df)
    census_continous_training.compute_census_continuous_training()


if __name__ == '__main__':
    ARGS = parse_args(args=sys.argv[1:])
    INPUT_DATA = ARGS.train_data
    EVAL_DATA = ARGS.eval_data
    df_train = read_file(INPUT_DATA, COLS)
    df_eval = read_file(EVAL_DATA, COLS)
    main(df_train, df_eval)
