try:
    from cayde.classifier import Classifier
except ImportError:
    from classifier import Classifier

try:
    from cayde.well.nlpwell import NLPWell
except ImportError:
    from well.nlpwell import NLPWell

import bert
import tensorflow as tf

from typing import List, Any, Optional, Dict

class BertClassifier(Classifier):
    _classifier: keras.Model

    def __init__(self, well: 'Well', *args, **kwargs):
        if not isinstance(well, NLPWell):
            raise TypeError("BERT Classifier only works with NLPWells")

        super().__init__(well, *args, **kwargs)

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
