import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from sortedcontainers import SortedDict


class TopRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, start_weight=0.1, confidence=0.97,
                 use_weights=True):
        self.decay = 1 - alpha
        self.start_weight = start_weight
        self.confidence = confidence
        self.use_weights = use_weights

    def fit(self, X, y):
        return self

    def predict(self, X):
        predictions = []
        for i in len(X):
            predictions.append(0)
        return predictions
    
    def _tag(self, job):
        """Each tag is a string formatted to contain the executable, user,
        user estimated run time, and number of required processors."""
        return '{}|{}|{}|{}'.format(
            job.executable_id, job.user_id, job.user_estimated_run_time, job.num_required_processors)

class Record(object):
    def __init__(self, start_value, start_weight, use_weights):
        self.dict = SortedDict()
        point_weight = start_weight*start_value if use_weights else start_weight
        self.dict[start_value] = (point_weight, 0)
        # Number of jobs in this class.
        self.count = 0
        self.t_pos = 0
        self.t_val = start_value
        self.over_weight = 0
        self.under_weight = 1

    def add(self, value, a_dec, threshold, use_weights):
        point_weight = value if use_weights else 1
        if value in self.dict:
            old_weight, old_count = self.dict[value]
            new_weight = point_weight + old_weight * \
                a_dec**(self.count-old_count)
        else:
            if value < self.t_val:
                self.t_pos += 1
            new_weight = point_weight
        self.dict[value] = (new_weight, self.count)
        self.over_weight *= a_dec
        self.under_weight *= a_dec
        # update over/under weight sums and move the threshold position if needed
        if value > self.t_val:
            self.over_weight += point_weight
            while self.under_weight / (self.under_weight + self.over_weight) <= threshold:
                self.t_pos += 1
                self.t_val = self.dict.peekitem(self.t_pos)[0]
                t_weight = self._update_t_weight(a_dec)
                self.under_weight += t_weight
                self.over_weight -= t_weight
                if self.t_pos == self.count + 1:
                    # set over_weight to zero when we got to the highest point
                    # to combat error accumulation.
                    # print("WARNING: predictor_top_percent: over_weight is reset")
                    self.over_weight = 0

        else:
            self.under_weight += point_weight
            if value < self.t_val:
                t_weight = self._update_t_weight(a_dec)
                while (self.under_weight - t_weight) / \
                      (self.under_weight + self.over_weight) > threshold:
                    self.under_weight -= t_weight
                    self.over_weight += t_weight
                    self.t_pos -= 1
                    self.t_val = self.dict.peekitem(self.t_pos)[0]
                    t_weight = self._update_t_weight(a_dec)
                    if self.t_pos == 0:
                        # reset under_weight to combat error accumulation
                        # print("WARNING: predictor_top_percent: under_weight is reset")
                        self.under_weight = t_weight
        self.count += 1

    def _update_t_weight(self, a_dec):
        old_weight, old_count = self.dict[self.t_val]
        new_weight = old_weight * a_dec ** (self.count - old_count)
        # print('old : ({}, {}), new: ({}, {})'.
        #     format(old_weight, old_count, new_weight, self.count))
        self.dict[self.t_val] = (new_weight, self.count)
        return new_weight