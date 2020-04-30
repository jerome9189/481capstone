from typing import Optional, List, Callable
import numpy as np
import time
import sys

def timeit(method):
    """
    Simple method decorator that tries to find bottle necks in the code.
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def progressBar(end: int, barWidth: int = 50, strCallback: Callable = lambda: ""):
    for i, _ in enumerate(range(end)):
        j = (i + 1) / end
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'=' * int(barWidth * j):{barWidth}s}] {int(100 * j)}% {strCallback()}")
        sys.stdout.flush()
        yield

class KFolds(object):

    def __init__(self,
        k_splits: int = None,
        training_ratios: List[float] = [1.0]
    ):
        '''K-Folds Cross Validator, where we follow the spirit of the
        sklearn API.

        Provides train/test indices to split data in train/test sets. Split
        dataset into k consecutive folds without shuffling.

        Each fold is then used once as a validation while the k - 1 remaining
        folds form the training set.

        Parameters
        ----------
            k_splits : int, default=None
                Number of splits to perform on the data. Default behavior is
                `k_splits` = number of data points.

            training_ratios : List of floats, default=[1.0]
                Percentage of training data to use when returning indexes.

        '''
        assert type(k_splits) == int, 'k_splits must be an integer'
        for training_ratio in training_ratios:
            assert training_ratio > 0.0 and training_ratio <= 1.0, (
                'all values in training_ratios must be ∈ (0, 1]'
            )

        # "protected" python variables wrapped with accessor properties
        # because C# represent
        self._k_splits = k_splits
        self._training_ratios = sorted(training_ratios)

    # These properties were included because I debugged using this shit a lot

    @property
    def k_splits(self) -> Optional[int]:
        "Explains the number of splits this class will do."
        return self._k_splits

    @property
    def training_ratios(self) -> List[float]:
        'Explains what % of the data is suggested for training'
        return self._training_ratios

    # Splitting occurs here.
    def split(self, data: np.array):
        '''Generate indices to split data into training and test set.

        Parameters
        ----------
            data : array-like, where shape = (n_samples, n_features)
                All available data, where n_samples is the number of samples and
                n_features is the number of features. n_features is not
                required.

        Yields
        ------
            train : ndarray
                The unshuffled training indices for that split.

            test : ndarray
                The unshuffled testing indices for that split.

            ratio : float
                What percentage of the available training data was used.
        '''
        step_size = 1
        if self.k_splits is not None:
            step_size = data.shape[0] // self.k_splits
            k_splits = self.k_splits
        else:
            k_splits = data.shape[0]

        for i in range(k_splits):
            for training_ratio in self.training_ratios:
                skip = max(1/training_ratio, 1.0)

                a = np.arange(
                    0,
                    i * step_size,
                    skip,
                    dtype=int
                )

                b = np.arange(
                    (i + 1) * step_size,
                    data.shape[0],
                    skip,
                    dtype=int
                )

                yield (
                    np.concatenate((a, b), axis=0),
                    np.arange(i * step_size, (i + 1) * step_size),
                    training_ratio
                )