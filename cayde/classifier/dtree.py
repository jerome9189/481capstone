from sklearn import tree

from .classifier import Classifier

import numpy as np

from typing import List, Any, Optional, Dict

class DecisionTreeClassifier(Classifier):
    _classifier: tree.DecisionTreeClassifier
    _max_depth: Optional[int]

    def reset(self):
        if len(self.args) == 1:
            self._max_depth = self.args[0]
        else:
            self._max_depth = self.kwargs.get('max_depth')
        self._classifier = tree.DecisionTreeClassifier(max_depth=self._max_depth)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.reset()
        self._classifier = self._classifier.fit(X, y)
