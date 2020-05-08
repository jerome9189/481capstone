import pandas as pd
import numpy as np
import time
import io
import os.path
import pickle
from threading import Thread
import matplotlib.pyplot as plt

try:
    from cayde.well.exceptions import DryWellException
except ImportError:
    from well.exceptions import DryWellException

from typing import Optional, Dict, Any, List, Union, Tuple, Callable

from sklearn.model_selection import train_test_split

CACHE_LOCATION = '.cayde/'

class UnivariateWell(object):
    """
        A "well" of information, which retrieves and caches information for ML purposes.
        Essentially maintains an internal pandas DF and returns useful modified copies
        in each of the properties.
    """

    class LazyCell(object):
        """
            When certain values in the well take a long time to compute/use large
            amounts of memory (e.g. tokenization), it's possible to place the 
            operation into a "LazyCell," where the operation will not take place
            until the cell is executed by the Well's `executeLazyCell()` method.
        """

        operation: Callable
        args: Tuple[Any, ...]
        kwargs: Dict[str, Any]

        def __init__(self, operation: Callable, args: Tuple[Any, ...] = tuple(), kwargs: Dict[str, Any] = dict()):
            self.operation = operation
            self.args = args
            self.kwargs = kwargs

        def compute(self):
            return self.operation(*self.args, **self.kwargs)


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

    def splitCategoricalData(
        self, 
        column: str, 
        expectedCategories: List[str] = [],
        prefix: str = 'is_', 
        useInts: bool = False
    ) -> List[str]:
        "Converts a categorical data column into multiple binary columns"
        if self._df is None:
            raise DryWellException()

        avail_columns = set()

        for category in expectedCategories:
            column_name = f'{prefix}{column}_{category}'
            if useInts:
                self._df[column_name] = 0
            else:
                self._df[column_name] = False
            avail_columns.add(column_name)

        if column not in self.all_columns:
            raise ValueError(f'Column "{column}" not in Well columns')
        for category in self._df[column].unique():
            column_name = f'{prefix}{column}_{category}'
            self._df[column_name] = self._df[column] == category

            if useInts:
                self._df[column_name] = self._df[column_name].astype(int)

            avail_columns.add(column_name)

        return list(avail_columns)

    def enumerateCategoricalData(self, columns: List[str], prefix: str = 'numerical') -> Dict[str, Dict[Any, int]]:
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
            well._df = self._df.head(n=maxRows)
        else:
            well._df = self._df.sample(n=maxRows)

        return well

    def shuffle(self, reset_index: bool = False):
        "Shuffles the dataframe"
        if self._df is None:
            raise DryWellException()

        self._df = self._df.sample(frac=1)

    def executeLazyColumns(self, start_index: int = 0 , end_index: int = -1, threads: bool = True) -> 'Well':
        if end_index < 0:
            end_index = len(self)

        thread_pool: List[Thread] = []

        def singleThreadTask(batch_df_, col_):
            batch_df_[col_] = batch_df_.apply(lambda row: row[col_].compute(), axis=1)

        transposed_df = self.df.T

        batch_df = transposed_df[range(start_index, end_index)].T
        for col in self._lazy_cols:
            if col in batch_df.columns:
                if threads:
                    thread_pool.append(Thread(target=singleThreadTask, args=(batch_df, col)))
                    thread_pool[-1].start()
                else:
                    batch_df[col] = batch_df.apply(lambda row: row[col].compute(), axis=1)
        
        if threads:
            for thread in thread_pool:
                thread.join()

        well = self.copy()
        well._df = batch_df
        
        return well

    def chunkGenerator(self, chunk_size: int = 1024 * 20, threads: bool = True):
        start_index = 0
        end_index = chunk_size
        while start_index < len(self):
            yield self.executeLazyColumns(start_index, end_index, threads)
            start_index, end_index = end_index, min(end_index + chunk_size, len(self))

    def expandColumn(self, column: str) -> List[str]:
        "Takes in a column of lists and expands it to multiple columns"
        if self._df is None:
            raise DryWellException()

        expectedSize = len(self._df.reset_index()[column][0])

        newColumns = {
            f'{column}_{i}': [] for i in range(expectedSize)
        }

        for cell in self._df[column]:
            for index, item in enumerate(cell):
                newColumns[f'{column}_{index}'].append(item)
    
        for key, column in newColumns.items():
            self._df[key] = column

        return list(newColumns.keys())


Well = UnivariateWell