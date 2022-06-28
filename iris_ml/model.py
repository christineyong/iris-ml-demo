'''
Script for training and evaluating XGBoost model.
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)

    logreg = LogisticRegression(C=1e5, max_iter=400)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    f1_score_result = f1_score(y_test, y_pred, average='macro')

    print(f'F1 score of Logistic Regression model: {f1_score_result}')

    return f1_score_result


if __name__ == '__main__':
    train_eval_model()
