'''
Script containing data-related functions, classes and methods.
'''
from pathlib import Path

import numpy as np

from ..data import get_data
from ..helper import get_config
from ..train import train

def test_get_data():
    '''
    Test function for loading data.
    '''
    get_data()

def test_get_config():
    '''
    Test that configuration file can be successfully loaded.
    '''
    cfg = get_config()
    assert isinstance(cfg, dict), (
        f'Expected output type "dict" but got output type {type(cfg)}')


def test_model_logreg():
    '''
    Test function for training and evaluating model.
    '''
    train('logreg')    

def test_model_xgboost():
    '''
    Test function for training and evaluating model.
    '''
    train('xgboost')    