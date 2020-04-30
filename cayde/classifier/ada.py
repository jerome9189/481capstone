from sklearn.ensemble import AdaBoostClassifier
from .classifier import Classifier
from .dtree import DecisionTreeClassifier

import numpy as np

from typing import List, Any, Dict

__all__ = ['AdaBosterClassifier']

class AdaBoosterClassifier(Classifier):
    _adaBooster: AdaBoostClassifier
    _base_estimator: Classifier
    _classifier_kwargs: Dict[str, Any]
    _n_estimators: int

    def reset(self):
        if len(self.args) > 1:
            raise ValueError("Too many positional arguments")

        if len(self.args) == 1:
            self._base_estimator = self.args[0]
        else:
            self._base_estimator = self.kwargs.get('base_estimator', DecisionTreeClassifier)

        self._n_estimators = self.kwargs.get('n_estimators', 50)
        self._adaBooster = AdaBoostClassifier(self._base_estimator(self._well, **self.kwargs.get('kwargs', {}))._classifier, n_estimators=self._n_estimators, algorithm='SAMME')

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.reset()
        self._adaBooster = self._adaBooster.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._adaBooster.predict(X)
