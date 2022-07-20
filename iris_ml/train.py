from .helper import get_config
from .data import get_data
from .model import XGBoostModel, LogisticRegressionModel

import os
import sys
from time import time, localtime, strftime
import hashlib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split


# Parenthesis create new instances of each object
models_dict = {
    'xgboost': XGBoostModel(),
    'logreg': LogisticRegressionModel(),
}

def train(model_arch=str('xgboost')):
    '''
    Trains and evaluates an XGBoost model on the iris dataset.
    Args:
        model_arch: Name of model architecture as stored in models_dict
    Returns:
      Path at which trained model was saved
    '''

    # Get configs
    cfg = get_config()

    # Generate hash to represent run
    time_now = time()
    str_formatted_time = strftime("%Y-%m-%dT%H:%M:%S", localtime())
    run_hash = hashlib.sha1()
    run_hash.update(str(time_now).encode('utf-8'))
    hash_str = run_hash.hexdigest()[:10]

    # Configure logger
    run_path = (
        Path(os.path.dirname(__file__)).parent/cfg['save_runs_path']/hash_str
    )
    run_path.mkdir(parents=True)
    logs_path = (run_path/f'train.log')
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

    logger.info(f'Started new training run at time {str_formatted_time}')
    logger.info(f'Run hash: {hash_str}')
    logger.info(f'Run data will be saved at: {run_path}')

    # Get data
    logger.info('Getting data...')
    iris = get_data()
    X = iris.data
    y = iris.target
    logger.info('Splitting data...')
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    logger.info('Done.')

    # Get model
    logger.info(f'Getting model {model_arch}...')
    model = models_dict[model_arch]
    logger.debug(f'Loaded model: ' + str(model))

    # Train model
    logger.info(f'Training model {model_arch}...')
    model.train(X_train, y_train)

    # Save model
    logger.info(f'Saving model {model_arch}...')
    model.save(directory=run_path)

    # Evaluate model on test set
    logger.info(f'Evaluating model {model_arch}...')
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)
    
    logger.info(f'Training done.')


if __name__ == '__main__':
    train()