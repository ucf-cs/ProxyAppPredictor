import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TemplateRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        predictions = []
        for i in len(X):
            predictions.append(0)
        return predictions
