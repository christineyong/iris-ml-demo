'''
Script for training and evaluating XGBoost model.
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb

from .data import get_data


def train_eval_model():
    '''
    Trains and evaluates an XGBoost model on the iris dataset.
    Returns:
      f1_score of trained model.
    '''
    iris = get_data()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 3}  # the number of classes that exist in this datset
    num_round = 5  # the number of training iterations

    bst = xgb.train(param, dtrain, num_round)

    preds = bst.predict(dtest)
    y_pred = np.asarray([np.argmax(line) for line in preds])
    f1_score_result = f1_score(y_test, y_pred, average='macro')

    print(f'F1 score of XGBoost model: {f1_score_result}')

    return f1_score_result


if __name__ == '__main__':
    train_eval_model()
