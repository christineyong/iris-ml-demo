'''
Script containing data-related functions, classes and methods.
'''
from pathlib import Path

import numpy as np

from ..data import get_data
from ..helper import get_config
from ..model import train_eval_model

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


def test_model():
    '''
    Test function for training and evaluating model.
    '''
    assert isinstance(train_eval_model(),np.float64)
    