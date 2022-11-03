import numpy as np
from itertools import product
from sklearn.base import BaseEstimator, RegressorMixin

# High-level ideas:

# fit():
# for each job, X[i], add its influence to the Cartesian product of each field included and not included.
# Influence is cumulative.

UNUSED = "UNUSED_VARIABLE_IN_COMBO"
# 24 hours in seconds.
UNKNOWN_TIME = 0 #86400


class CompleteRegressor(BaseEstimator, RegressorMixin):
    # Get all combos of True/False for each feature.
    def get_tag_combos(self, X):
        num_features = len(X)
        feature_options = []
        # This for loop will naturally form combo lists from most specialized
        # to most generalized for arbitrary parameter sets.
        for i in range(num_features):
            feature_options.append([True, False])
        # Get all combos of True/False for each feature.
        combos = list(product(*feature_options))
        # DEBUG: Ensure the list actually is sorted from most specialized to 
        # most generalized.
        if len(combos) > 0:
            curr_count = num_features
            for val in combos:
                pred_count = curr_count
                curr_count = 0
                for x in val:
                    if x == True:
                        curr_count += 1
                assert curr_count <= pred_count
        return combos

    # UNUSED
    # Get the variance of the given combo's dictionary.
    def _var(self, dictionary):
        assert dictionary.count > 1, "need more than 1 measurement"
        var = (dictionary.square - dictionary.avg * dictionary.sum) \
            / (dictionary.count - 1)
        if var < 0:
            if var < -1:
                raise Exception()
            var = 0.0
        return var

    def __init__(self, columns, alpha=0.2):
        # The column names for this application.
        self.columns = columns
        self.val = 0
        # The head of the prediction trie.
        self.preds = {}
        # The remaining influence of previous predictions. Controls falloff.
        self.alpha = alpha
        # Does nothing. Formerly controlled strength of variance value.
        self.sigma_factor = None

    def fit(self, X, y):
        y = list(y)
        # print("X: "+str(X))
        # print("y: "+str(y))
        combos = self.get_tag_combos(X)
        # For each job.
        for job_index, job in enumerate(X):
            # For each True/False combo of tags.
            for combo in combos:
                # The head of the dict trie.
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
                    dict_curr.update({
                        "avg": value,
                        "square": value**2,
                        "count": 1,
                        "sum": value
                    })
                # Otherwise perform a weighted average of it with existing
                # run times of similar jobs.
                else:
                    dict_curr["count"] = 1 + (1 - self.alpha) * dict_curr["count"]
                    dict_curr["sum"] = value + (1 - self.alpha) * dict_curr["sum"]
                    dict_curr["avg"] = dict_curr["sum"] / dict_curr["count"]
                    dict_curr["square"] = value * value + \
                        (1 - self.alpha) * dict_curr["square"]
                    # var = self._var(dict_curr)
                    # return dict_curr["avg"] + self.sigma_factor * var**0.5
        return self

    # For each job, find the most closely related historic job by tag
    # similarity and use that as the predicted run time.
    def predict(self, X):
        combos = self.get_tag_combos(X)
        # The list of predictions corresponding to each job in X.
        # Must be built up and returned.
        predictions = []
        # For each job.
        for job_index, job in enumerate(X):
            found = False
            # For each True/False combo of tags.
            # Identify the most specialized group.
            # NOTE: The combo list is presorted from most specialized to
            # least specialized already.
            for combo in combos:
                # Check for a match.
                # Start at the head of the trie.
                dict_curr = self.preds
                # Assume a valid match unless found to be invalid.
                valid = True
                # For each tag in use (or disuse).
                # Traverse down the trie, looking for the best match.
                for use_index, use in enumerate(combo):
                    # The value associated with this index.
                    key = UNUSED if not use else job[use_index]
                    # Avoid groups that don't match.
                    if key not in dict_curr.keys():
                        valid = False
                        break
                    # Continue traversing down.
                    dict_curr = dict_curr[key]
                # If the tag trie traversal resulted in finding a valid leaf.
                if valid:
                    # Get the predicted run time to this combo.
                    predictions.append(dict_curr["avg"])
                    found = True
                    # Stop looking for more combos.
                    break
            # If no matching combo was found.
            if not found:
                # Then there is no historic time available to predict.
                # Use the fallback time.
                predictions.append(UNKNOWN_TIME)
        return predictions
    
    def predict_and_fit(self, x, y):
        prediction = self.predict([x])
        self.fit([x],[y])
        return prediction[0]
