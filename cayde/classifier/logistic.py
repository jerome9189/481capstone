from sklearn.linear_model import LogisticRegression

from .classifier import Classifier

import numpy as np

from typing import List, Any, Optional, Dict

class LogisticClassifier(Classifier):
    _classifier: LogisticRegression
    _C: Optional[int]
    _threshold: float

    def reset(self):
        if len(self.args) >= 1:
            self._C = self.args[0]
        else:
            self._C = self.kwargs.get('C')
        if len(self.args) >= 2:
            self._threshold = self.args[1]
        else:
            self._threshold = self.kwargs.get('threshold', 0.5)
        self._classifier = LogisticRegression(C=self._C)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.reset()
        self._tree = self._classifier.fit(X, y)

    def predict(self, input_cols: List[Any]):
        return self._classifier.predict(input_cols) > self._threshold

