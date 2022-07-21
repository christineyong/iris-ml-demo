'''
Script for training and evaluating XGBoost model.
'''
import os
import pickle
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score

from .helper import get_config


logger = logging.getLogger(__name__)

def f1_score_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def fpointfive_score_micro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5, average='micro')

def f2_score_micro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='micro')

class IrisModel():
    def __init__(self, model_base=None):
        self.model_base = model_base
        self.name = None
        self.eval_metrics = {
            'accuracy': accuracy_score,
            'f1_score': f1_score_micro,
            'f0.5_score': fpointfive_score_micro,
            'f2_score': f2_score_micro,
        }
        self.param = {}

    def preprocess():
        return

    def train(self):
        logger.info(f'Model: {self.model_base} - {self.name}')
        logger.info(f'Parameters:')
        for param, param_value in self.param.items():
            logger.info(f'\t{param}:\t{param_value}')

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        y_pred = np.asarray([np.argmax(line) for line in predictions])
        return y_pred

    def save(self, directory):
        '''
        Save model as a pickle file.
        Args:
            directory: Directory in which to save model.
        '''
        model_name = f'{self.model_base}_{self.name}'
        logging.debug(f'Model artifact save directory: {directory}')
        if not directory.exists():
            logger.warning(f'Model artifact directory does not exist.')
            logger.warning(f'Creating directory at {model_pickle_dir}...')
            directory.mkdir(parents=True)
        model_pickle_path = (directory/model_name).with_suffix('.pickle')
        model_pickle_path.touch()
        logger.info(f'Saving model to: {model_pickle_path}...')
        with open(model_pickle_path, 'wb') as pickle_path:
            pickle.dump(self.model, pickle_path, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'Model saved.')
        
    def load(self, directory):
        '''
        Load model from a pickle file
        Args:
            directory: Directory of the model pickle file
        '''
        model_name = f'{self.model_base}_{self.name}'
        logging.debug(f'Model artifact read directory: {directory}')
        if not directory.exists():
            logger.warning(f'Model artifact directory does not exist.')
            directory.mkdir(parents=True)
        model_pickle_path = (directory/model_name).with_suffix('.pickle')
        model_pickle_path.touch()
        logger.info(f'Reading model from: {model_pickle_path}...')
        with open(model_pickle_path, 'rb') as pickle_path:
            self.model = pickle.load(pickle_path)
        logger.info(f'Model loaded.')
        
    def evaluate(self,y_true, y_pred):
        logger.info('Evaluation metrics:')
        for metric, fmetric in self.eval_metrics.items():
            metric_value = fmetric(y_true, y_pred)
            logger.info(f'\t{metric}:\t{metric_value}')
        

class XGBoostModel(IrisModel):
    def __init__(self):
        super().__init__(model_base='XGBoost')
        self.name = 'Vanilla'
        self.param = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 3, # the number of classes that exist in this datset
        }  
        self.max_iterations = 5  # the number of training iterations

    def preprocess(self, X, y):
        return xgb.DMatrix(X, label=y)

    def train(self, X_train, y_train):
        super().train()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(
            self.param,
            dtrain,
            self.max_iterations,
        )

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)
        y_pred = np.asarray([np.argmax(line) for line in predictions])
        return y_pred


class LogisticRegressionModel(IrisModel):
    def __init__(self):
        super().__init__(model_base='Logistic Regression')
        self.name = 'Vanilla'
        self.param = {
            'C': 1e5,
            'max_iter': 400
        }

    def train(self, X_train, y_train):
        super().train()
        self.model = LogisticRegression(**self.param)
        self.model.fit(X_train, y_train)    