import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ReqtimeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def predict(self, X):
        predictions = []
        for job in X:
            # NOTE: This data isn't currently passed because it isn't unique.
            # Manually add it.
            predictions.append(86400)
        return predictions
    
    def predict_and_fit(self, x, y):
        prediction = self.predict([x])
        self.fit([x],[y])
        return prediction[0]
