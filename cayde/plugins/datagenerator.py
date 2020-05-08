import numpy as np
import tensorflow.keras as keras
from cayde.well import Well

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, well: Well, columnsToExpand: str = '', batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.well = well
        self.shuffle = shuffle
        self.expandColumns = columnsToExpand
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.well) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        well = self.well.executeLazyColumns(index * self.batch_size, (index + 1) * self.batch_size)

        if self.expandColumns:
            well.input_cols = well.expandColumn(self.expandColumns)

        X = well.df[well.input_cols].to_numpy().T
        y = well.df[well.output_col].to_numpy().T

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.well.shuffle()