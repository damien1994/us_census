"""
Python script to define logger config
author: Damien Michelle
date: 03/09/2021
"""
import os
import logging

logging.basicConfig(
            filename=os.path.join(os.path.dirname(__file__), 'logs/census_predictions.log'),
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')
