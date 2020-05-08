import pandas as pd
import numpy as np
import time
import io
import os.path
import pickle
import matplotlib.pyplot as plt

try:
    from cayde.well.exceptions import DryWellException
except ImportError:
    from well.exceptions import DryWellException

from typing import Optional, Dict, Any, List, Union, Tuple

from sklearn.model_selection import train_test_split

CACHE_LOCATION = '.cayde/'

class UnivariateWell(object):
    """
        A "well" of information, which retrieves and caches information for ML purposes.
        Essentially maintains an internal pandas DF and returns useful modified copies
        in each of the properties.
    """
    name: str = 'base_well'
    version: float = 0.01

    _last_retrieved: int
    _df: Optional[pd.DataFrame]
    _source: str

    _input_cols: List[str]
    _output_col: Optional[str]
    _lazy_cols: List[str]
    _feature_shape: Tuple[int, ...]

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        "Returns the shape of an instance feature"
        return self._feature_shape

    @feature_shape.setter
    def feature_shape(self, newShape: Tuple[int, ...]):
        if np.prod(self.X.shape[1:]) != np.prod(newShape):
            raise ValueError("Proposed shape does not fit existing data")
        self._feature_shape = newShape

    @property
    def input_cols(self) -> List[str]:
        if self._df is None:
            raise DryWellException()
        return self._input_cols[:]

    @input_cols.setter
    def input_cols(self, newColumns: List[str]):
        if self._df is None:
            raise DryWellException()
        for column in newColumns:
            if column not in self._df:
                raise ValueError(f"{column} not found in well")
        self._feature_shape = (len(newColumns),)
        self._input_cols = newColumns[:]

    @property
    def output_col(self) -> Optional[str]:
        if self._df is None:
            raise DryWellException()
        return self._output_col

    @output_col.setter
    def output_col(self, newColumn: str):
        if self._df is None:
            raise DryWellException()
        if newColumn not in self._df:
            raise ValueError(f"{newColumn} not found in well")
        self._output_col = newColumn

    @property
    def lazy_cols(self) -> List[str]:
        if self._df is None:
            raise DryWellException()
        return self._lazy_cols

    @lazy_cols.setter
    def lazy_cols(self, newColumns: List[str]):
        if self._df is None:
            raise DryWellException()
        for column in newColumns:
            if column not in self._df:
                raise ValueError(f"{column} not found in well")
        self._lazy_cols = newColumns[:]

    @property
    def all_columns(self) -> List[str]:
        if self._df is None:
            raise DryWellException()
        return self._df.columns

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise DryWellException()
        df = self._df.copy()
        df.drop(columns=filter(lambda c: c not in self._input_cols and c != self._output_col, df.columns), inplace=True)
        return df

    @property
    def input_df(self) -> pd.DataFrame:
        if self._df is None:
            raise DryWellException()
        if not self._input_cols:
            raise ValueError("No Inputs Specified")
        return pd.DataFrame(data={
            key: self._df[key] for key in self._input_cols
        })

    @property
    def X(self) -> np.ndarray:
        X = self.input_df.to_numpy()
        return X.reshape((X.shape[0],) + self._feature_shape)

    @property
    def output_df(self) -> pd.DataFrame:
        if self._df is None:
            raise DryWellException()
        if self._output_col is None:
            raise ValueError("No Output Specified")
        return pd.DataFrame(data={
            self._output_col: self._df[self._output_col]
        })

    @property
    def output_series(self) -> pd.DataFrame:
        if self._df is None:
            raise DryWellException()
        if self._output_col is None:
            raise ValueError("No Output Specified")
        return self.df[self._output_col]

    @property
    def y(self) -> np.ndarray:
        return self.output_df.to_numpy()

    @property
    def last_retrieved(self) -> int:
        return self._last_retrieved

    @property
    def dry(self) -> bool:
        return self._df is None

    def __init__(self, source: str, *args, **kwargs):
        self._last_retrieved = -1
        self._df = None
        self._input_cols = []
        self._lazy_cols = []
        self._output_col = None
        self._source = source
        self._feature_shape = (1,)

    def __len__(self):
        if self._df is None:
            return -1
        return len(self._df)

    def splitTrainingTesting(self, testingRatio: float = 0.1) -> Tuple['Well', 'Well']:
        training_df, testing_df = train_test_split(self._df, test_size = testingRatio)
        training_well = self.copy()
        training_well._df = training_df
        testing_well = self.copy()
        testing_well._df = testing_df

        return training_well, testing_well

    def splitCategoricalData(self, columns: List[str], prefix: str = 'is_'):
        "Converts a categorical data column into multiple binary columns"
        if self._df is None:
            raise DryWellException()
        for column in columns:
            if column not in self.all_columns:
                raise ValueError(f'Column "{column}" not in Well columns')
            for category in self._df[column].unique():
                self._df[f'{prefix}{column}_{category}'] = self._df[column] == category

    def enumerateCategoricalData(self, columns: List[str], prefix: str = 'onehot') -> Dict[str, Dict[Any, int]]:
        "Converts columns containing categorical data into one-hot encodings"
        if self._df is None:
            raise DryWellException()

        atlas: Dict[str, Dict[Any, int]] = {}

        for column in columns:
            if column not in self.all_columns:
                raise ValueError(f'Column "{column}" not in Well columns')

            atlas[column] = {}
            self._df[f'{prefix}_{column}'] = -1

            for index, category in enumerate(self._df[column].unique()):
                self._df[f'{prefix}_{column}'][self._df[column] == category] = index
                atlas[column][category] = index

        return atlas

    def prune(self, saveColumns: List[str] = list()) -> List[str]:
        "Prunes the internal dataframe of this well"
        if self._df is None:
            raise DryWellException()

        columns = self._df.columns
        deleted_columns: List[str] = []
        for column in columns:
            if column not in self.input_cols and column not in saveColumns and column != self.output_col:
                del self._df[column]
                deleted_columns.append(column)

        return deleted_columns

    def fetch(self):
        if self._source.endswith('.csv'):
            self._df = pd.read_csv(self._source)
        elif self._source.endswith('.dat'):
            self._df = pd.read_csv(self._source, header=None)
            self._df.columns = [f'col_{i}' for i in self._df.columns]
        elif self._source.endswith('.xlsx'):
            self._df = pd.read_excel(self._source)
        else:
            raise ValueError(f"Unknown file type: {self._source}")
        self._last_retrieved = int(time.time())

    def save(self):
        if not os.path.isdir(CACHE_LOCATION):
            os.mkdir(CACHE_LOCATION)
        with io.open(os.path.join(CACHE_LOCATION, f"{self.name}.data"), "wb") as handle:
            handle.write(pickle.dumps({
                'version': self.version,
                'name': self.name,
                'last_retrieved': self._last_retrieved,
                'df': self._df,
                'input_cols': self.input_cols,
                'output_col': self.output_col,
                'lazy_cols': self.lazy_cols,
                'feature_shape': self._feature_shape
            }))

    def load(self):
        with io.open(os.path.join(CACHE_LOCATION, f'{self.name}.data'), 'rb') as handle:
            result: Dict[str, Any] = pickle.loads(handle.read())

        if result.get('name') != self.name:
            raise ValueError(f"Cached data uses different well type ({self.name} vs {result.get('name')})")

        if result.get('version', -1) != self.version:
            raise ValueError("Cached data uses different well version")

        self._df = result['df']
        if result['input_cols'] is not None:
            self.input_cols = result['input_cols']
        if result['output_col'] is not None:
            self.output_col = result['output_col']
        self._lazy_cols = result['lazy_cols']
        self._last_retrieved = result['last_retrieved']
        self._feature_shape = result['feature_shape']

    def load_or_fetch(self) -> bool:
        "Attempt to load the data, and if it fails, fetch and save it. Returns True if load was succesful, false if fetch was required."
        try:
            self.load()
            return True
        except (FileNotFoundError) as e:
            self.fetch()
            self.save()
            return False

    def describe(self):
        return self._df.describe()

    def copy(self) -> 'Well':
        well = self.__class__(self._source)

        if self._df is not None:
            well._df = self._df.copy()

        well._last_retrieved = self._last_retrieved
        well._input_cols = self._input_cols[:]
        well._output_col = self._output_col
        well._lazy_cols = self._lazy_cols[:]
        well._source = self._source[:]
        well._feature_shape = self._feature_shape

        return well

    def getToySample(self, maxRows: int = 20, useHead: bool = False) -> 'Well':
        well = self.copy()

        if self._df is None:
            raise DryWellException()

        if useHead:
            well._df = self._df.head(maxRows)
        else:
            well._df = self._df.sample(n=maxRows)

        return well

    def shuffle(self, reset_index: bool = False):
        "Shuffles the dataframe"

    def batchGenerator(self, batch_size: int = 32):
        
        # Get batch_size elements at a time
        # Populate the "_lazy_cols" upon generation
        # yield a copy of the dataframe

        def generator() -> pd.DataFrame:
            start_index = 0
            end_index = batch_size
            # while start_index < :
            #     yield None

        return generator

Well = UnivariateWell