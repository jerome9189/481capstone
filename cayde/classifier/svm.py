from sklearn import svm as sklearnsvm

from .classifier import Classifier

import numpy as np

from typing import List, Any, Optional, Dict

class SVMClassifier(Classifier):
    _classifier: sklearnsvm.SVC
    _C: Optional[int]

    def reset(self):
        if len(self.args) == 1:
            self._C = self.args[0]
        else:
            self._C = self.kwargs.get('C', 1.0)
        self._classifier = sklearnsvm.SVC(C=self._C)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.reset()
        self._classifier = self._classifier.fit(X, y)

