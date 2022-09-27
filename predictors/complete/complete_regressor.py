import numpy as np
from itertools import product
from sklearn.base import BaseEstimator, RegressorMixin

# High-level ideas:

# fit():
# for each job, X[i], add its influence to the Cartesian product of each field included and not included.
# Influence is cumulative.

UNUSED = "UNUSED_VARIABLE_IN_COMBO"
# 24 hours in seconds.
UNKNOWN_TIME = 86400


class CompleteRegressor(BaseEstimator, RegressorMixin):
    def get_tag_combos(self, X):
        num_features = X.shape[1]
        feature_options = []
        # This for loop will naturally form combo lists from most specialized
        # to most generalized for arbitrary parameter sets.
        for i in range(num_features):
            feature_options.append([True, False])
        combos = list(product(*feature_options))
        return combos

    def _var(self, dictionary):
        assert dictionary.count > 1, "need more than 1 measurement"
        var = (dictionary.square - dictionary.avg * dictionary.sum) \
            / (dictionary.count - 1)
        if var < 0:
            if var < -1:
                raise Exception()
            var = 0.0
        return var

    def __init__(self, alpha=0.2):
        self.val = 0
        self.preds = {}
        self.alpha = alpha
        self.sigma_factor = None

    def fit(self, X, y):
        # print("X: "+str(X))
        # print("y: "+str(y))
        combos = self.get_tag_combos(X)
        # For each job.
        for job_index, job in enumerate(X):
            # For each True/False combo of tags.
            for combo in combos:
                dict_curr = self.preds
                # For each tag use (or disuse).
                for use_index, use in enumerate(combo):
                    # Get the name of the parameter to use as a key, if used.
                    key = UNUSED if not use else job[use_index]
                    # Build up a trie of parameters.
                    if key not in dict_curr.keys():
                        dict_curr[key] = {}
                    dict_curr = dict_curr[key]
                # Now we are at the bottom of our trie.

                # Assign the run time to this combo.
                value = y[job_index]

                # Initialize the ending dictionary entry.
                # If no entry exists, create a new one.
                if len(dict_curr) == 0:
                    dict_curr = {
                        "avg": value,
                        "square": value**2,
                        "count": 1,
                        "sum": value
                    }
                # Otherwise perform a weighted average of it with existing
                # run times of similar jobs.
                else:
                    dict_curr.count = 1 + (1 - self.alpha) * dict_curr.count
                    dict_curr.sum = value + (1 - self.alpha) * dict_curr.sum
                    dict_curr.avg = dict_curr.sum / dict_curr.count
                    dict_curr.square = value*value + \
                        (1 - self.alpha) * dict_curr.square
                    # var = self._var(dict_curr)
                    # return dict_curr.avg + self.sigma_factor * var**0.5
        return self

    def predict(self, X):
        combos = self.get_tag_combos(X)

        predictions = []

        # For each job
        for job_index, job in enumerate(X):
            found = False
            # For each True/False combo of tags.
            # Identify the most specialized group.
            for combo in combos:
                # Check for a match.
                dict_curr = self.preds
                valid = True
                # For each tag use (or disuse).
                for use_index, use in enumerate(combo):
                    key = UNUSED if not use else job[use_index]
                    # Avoid groups that don't match.
                    if key not in dict_curr.keys():
                        valid = False
                        break
                    dict_curr = dict_curr[key]
                # If the tag trie traversal resulted in finding a valid leaf.
                if valid:
                    # Get the predicted run time to this combo.
                    predictions.append(dict_curr.avg)
                    found = True
                    break
            if not found:
                predictions.append(UNKNOWN_TIME)
        return predictions
