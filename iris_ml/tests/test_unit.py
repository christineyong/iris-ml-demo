'''
Script containing data-related functions, classes and methods.
'''
import numpy as np

from ..data import get_data
from ..model import train_eval_model

def test_get_data():
    '''
    Test function for loading data.
    '''
    get_data()

def test_model():
    '''
    Test function for training and evaluating model.
    '''
    assert isinstance(train_eval_model(),np.float64)
    