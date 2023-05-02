import numpy as np
from sklearn.model_selection._split import _BaseKFold


class RollingOriginForwardValidation(_BaseKFold):
    """
    Parameters
    ----------
    min_train_size: int
        minimum number of time units to include in each train set
        default is 50
    test_size: int
        number of time units to include in each test set
        default is 4
    """

    def __init__(self, min_train_size=50, test_size=6):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.n_splits = None  # will be adjusted dynamically during split()
        super().__init__(n_splits=2, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None, gap=0):
        """
        Generate indices to split data into training and test set

        Parameters
        ----------
        X: pandas DataFrame
            training data

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        n_folds = (n_samples - self.min_train_size) // self.test_size
        self.n_splits = n_folds - 1

        test_starts = range(
            n_samples - self.test_size * n_folds, n_samples, self.test_size
        )

        for test_start in test_starts:
            train_end = test_start - gap - 1
            yield (
                indices[: train_end + 1],
                indices[test_start : test_start + self.test_size],
            )


# https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
class BlockingTimeSeriesSplit:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start:mid], indices[mid + margin : stop]
