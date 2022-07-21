from .helper import get_config
from .data import get_data
from .model import XGBoostModel, LogisticRegressionModel

import os
import sys
from time import time, localtime, strftime
import hashlib
import logging
from pathlib import Path

import pandas as pd


# Parenthesis create new instances of each object
models_dict = {
    'xgboost': XGBoostModel(),
    'logreg': LogisticRegressionModel(),
}

def predict(model_arch=str('xgboost')):
    '''
    Predict on the iris dataset.
    Args:
        model_arch: Name of model architecture as stored in models_dict
    Returns:
      ...
    '''

    # Get configs
    cfg = get_config()
    time_now = time()
    str_formatted_time = strftime("%Y-%m-%dT%H:%M:%S", localtime())
    run_hash = hashlib.sha1()
    run_hash.update(str(time_now).encode('utf-8'))
    hash_str = run_hash.hexdigest()[:10]
    # Configure logger
    run_path = (
        Path(os.path.dirname(__file__)).parent/cfg['pretrain_model_path']
    )
    run_path.mkdir(parents=True)
    logs_path = (
        Path(os.path.dirname(__file__)).parent/cfg['pretrain_model_path']/hash_str
    )
    logs_path.mkdir(parents=True)
    logs_path = (logs_path/f'predict.log')
    logs_path.touch(exist_ok=False)
    
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(logs_path.absolute()), # Save to file
            logging.StreamHandler(), # Output to terminal
        ],
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info(f'Load model at time {str_formatted_time}')
    logger.info(f'Model is loaded from: {run_path}')
    
    # Get data
    logger.info('Getting data for model to predict...')
    iris = get_data()
    ### For testing purpose only (test data subset of train data) ###
    X = iris.data[:100]
    y = iris.target[:100]
    
    # Get model
    logger.info(f'Getting model {model_arch}...')
    model = models_dict[model_arch]
    logger.debug(f'Loaded model: ' + str(model))    
    
    # Load model
    logger.info(f'Saving model {model_arch}...')
    model.load(directory=run_path)

    # predict data
    prediction_path = (
        Path(os.path.dirname(__file__)).parent/cfg['prediction_path']
    )
    prediction_path.mkdir(parents=True)    
    y_pred = model.predict(X)   
    X = pd.DataFrame(X)
    X['predicted'] = y_pred
    X.to_csv(prediction_path/f'predicted.csv')
    logger.info(f'Saving predicted data to {prediction_path}')
    
    logger.info(f'Prediction done.')


if __name__ == '__main__':
    predict()    
    
    
    