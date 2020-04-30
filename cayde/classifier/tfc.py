try:
    from cayde.classifier import Classifier
except ImportError:
    from classifier import Classifier

import numpy as np
import tensorflow as tf
from tensorflow import keras

from typing import List, Any, Optional, Dict

class TensorflowClassifier(Classifier):
    _classifier: keras.Model

    def reset(self):
        if len(self.args) == 1 and isinstance(self.args[0], keras.Model):
            self._classifier = self.args[0]
        else:
            raise ValueError("Second positional argument must be a keras model")
        self._classifier.compile(
            optimizer=self.kwargs.get('optimizer', 'adam'),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=self.kwargs.get('metrics', ['accuracy'])
        )

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.reset()
        self._classifier.fit(X, y, epochs=self.kwargs.get('epochs', 10))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._classifier.predict(X), axis=1)

    def summary(self) -> str:
        return self._classifier.summary()
