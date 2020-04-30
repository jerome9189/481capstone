import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

try:
    from cayde.well import Well # type: ignore
except ImportError:
    from well import Well # type: ignore
from typing import List, Any, Dict, Union, Tuple

class Classifier(object):
    _well: Well
    _classifier: Any
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    class InvalidWellException(Exception):
        "An exception thrown when a well is incompatibile with this classifier"

    def __init__(self, well: Well, *args, **kwargs):
        self._well = well
        self.args = args
        self.kwargs = kwargs
        if self._well.output_series.dtype == float:
            raise self.InvalidWellException("output of well is continuous data")
        self.reset()

    @property
    def possibleClassifications(self):
        return len(self._well.output_series.unique())

    def reset(self):
        "Override this method"

    def fit(self):
        self._fit(self._well.X, self._well.y)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X)

    def accuracy(self, k_folds: int = 3, verbose: bool = False) -> pd.DataFrame:
        """
        Returns a dataframe describing how accurate this classifier is.

        Paticiularly useful when called with `classifier.accuracy().mean()`,
        which shows the average scores across kfolds. Also useful is
        `classifier.accuracy().mean()['precision'] - classifier._well.y.mean()`,
        which tells you how much better the classifier performs than only mentioning
        one answer. A good metric for highly imbalanced data.
        """
        X = self._well.X
        y = self._well.y

        isBinary = self.possibleClassifications == 2


        kf = KFold(n_splits=k_folds)
        accuracies: Dict[str, List[float]] = {
            'f1': [],
            'accuracy': [],
            'recall': [],
            'precision': [],
        }

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            if verbose: print(f'Kfolds Iteration: {i + 1}/{k_folds}')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # train the decision tree and output predictions on the remaining data
            self._fit(X_train, np.ravel(y_train))
            y_pred = self.predict(X_test)

            accuracies['accuracy'].append(accuracy_score(y_test, y_pred))
            if isBinary:
                accuracies['f1'].append(f1_score(y_test, y_pred))
                accuracies['recall'].append(recall_score(y_test, y_pred))
                accuracies['precision'].append(precision_score(y_test, y_pred))

        df = pd.DataFrame(data={k: v for (k, v) in accuracies.items() if len(v)})
        if isBinary:
            df.transpose(copy=True)
            df['accuracyImprovement'] = df['accuracy'] - max(float(self._well.output_df.mean()), 1 - float(self._well.output_df.mean()))
        return df