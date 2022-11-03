from sklearn.base import BaseEstimator, RegressorMixin


class TsafrirRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, columns):
        self.run_time_prev = None
        self.run_time_curr = None
        self.predict_multiplier = 1 #multiplier
        self.columns = columns

    # NOTE: Must run after each job completes.
    # NOTE: Should never fit more than one job at a time.
    def fit(self, X, y):
        y = list(y)
        if len (y) < 1:
            pass
        elif len(y) == 1:
            self.run_time_prev = self.run_time_curr
            self.run_time_curr = y[-1]
        elif len(y) >= 2:
            self.run_time_prev = y[-2]
            self.run_time_curr = y[-1]
        return self

    # NOTE: Should never predict more than one job at a time.
    def predict(self, X):
        predictions = []
        for job in X:
            # NOTE: This data isn't currently passed because it isn't unique.
            # Manually add it.
            user_estimated_run_time = 0 #86400
            # if job.user_id not in self.run_time_prev:
            #     self.run_time_prev = None
            #     self.run_time_prev = None
            if self.run_time_prev is not None:
                average = (self.run_time_curr + self.run_time_prev) / 2
                predicted_run_time = min(user_estimated_run_time, average)
            else:
                # Fallback
                predicted_run_time = user_estimated_run_time
            predictions.append(predicted_run_time * self.predict_multiplier)
        return predictions

    def predict_and_fit(self, x, y):
        prediction = self.predict([x])
        self.fit([x],[y])
        return prediction[0]