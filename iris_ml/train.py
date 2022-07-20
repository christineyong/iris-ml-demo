from .data import get_data
from .model import XGBoostModel, LogisticRegressionModel

import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

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
    logger.info('Started new training run.')
    logger.info('Getting data...')
    iris = get_data()
    X = iris.data
    y = iris.target
    logger.info('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    logger.info('Done.')

    logger.info(f'Getting model {model_arch}...')
    model = models_dict[model_arch]
    logger.debug(f'Got model: ' + str(model))

    logger.info(f'Training model {model_arch}...')
    model.train(X_train, y_train)

    logger.info(f'Saving model {model_arch}...')
    model.save()

    logger.info(f'Evaluating model {model_arch}...')
    y_pred = model.predict(X_test)
    model.evaluate(y_test, y_pred)
    
    logger.info(f'Training done.')


if __name__ == '__main__':
    train()