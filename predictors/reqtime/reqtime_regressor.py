import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ReqtimeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        predictions = []
        for i in len(X):
            try:
                predictions.append(X["reqtime"][i])
            except KeyError:
                predictions.append(86400)
        return predictions
