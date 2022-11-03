import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

# Notes.
# Use known run time equation.
# Learn constants for specific machine.
# Logarithmic transformation may help solve exponent factors:
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
# SWFFT: n_repetitions * ngx * ngy * ngz


class AnalyticalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, columns):
        self.columns = columns

    # NOTE: Must run after each job completes.
    # NOTE: Should never fit more than one job at a time.
    def fit(self, X, y):
        return self

    # NOTE: Should never predict more than one job at a time.
    def predict(self, X):
        if X.shape[1] != 6 or \
                self.columns != ["n_repetitions", "ngx", "ngy", "ngz",
                                 "nodes", "tasks"]:
            # Fallback for non-SWFFT code.
            print("Warning: Non-SWFFT code passed to analytical regressor.")
            predictions = []
            for i in X.shape[0]:
                predictions.append(86400)
            return predictions

        # Column labels are not normally passed in. Recover them.
        df = pd.DataFrame(X, columns=["n_repetitions", "ngx", "ngy", "ngz",
                                      "nodes", "tasks"])
        # DEBUG
        # print(df)

        # Multiplier found on Voltrino to best match scaling to seconds.
        MULTIPLIER = 1/4349848.332
        # These factors and relationships were found by visually inspecting
        # correlation plots. ML
        predictions = df["n_repetitions"] * \
            df["ngx"] * \
            df["ngy"] * \
            df["ngz"] * \
            pow(df["nodes"], -0.8) * \
            pow(df["tasks"], -0.7) * \
            MULTIPLIER
        # DEBUG
        # print(predictions)
        return predictions

    def predict_and_fit(self, x, y):
        prediction = self.predict([x])
        return prediction
