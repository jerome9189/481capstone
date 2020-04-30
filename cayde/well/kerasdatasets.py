from cayde.well import Well
from keras import datasets

from cayde.well.exceptions import WellCacheException

import pandas as pd
import numpy as np

__all__ = ["FashionWell", "KerasDatsetWell"]

class KerasDatasetWell(Well):
    name = "KerasWell"

    def fetch(self):
        resource = getattr(datasets, self._source)
        (train_images, train_labels), (test_images, test_labels) = resource.load_data()
        if len(train_images.shape) >= 3:
            axis1_dim = train_images.shape[1]
            for dim in train_images.shape[2:]:
                axis1_dim *= dim
            train_images = train_images.reshape(train_images.shape[0], axis1_dim)
            test_images = test_images.reshape(test_images.shape[0], axis1_dim)
        self._df = pd.DataFrame(data=train_images)
        self._df.columns = self.input_cols = [f'col_{i}' for i in self._df.columns]
        self._df['y'] = train_labels
        self.output_col = 'y'

class FashionWell(KerasDatasetWell):

    def __init__(self):
        super().__init__('fashion_mnist')
        self._feature_shape = (28, 28)