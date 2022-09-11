from itertools import chain

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable
from sklearn.utils import _safe_indexing, check_random_state

from ..data import Data


class KFold(_BaseKFold):
    '''
    KFold splitter which works on a naplib.Data object or a list-like sequence.
    
    Parameters
    ----------
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        
    Examples
    --------
    >>> from naplib.model_selection import KFold
    >>> list1 = [1,2,3] # this could be a field of a Data object, like data['resp']
    >>> list2 = [5,6,7] # this could be another field of a Data object, like data['aud']
    >>> kfold = KFold(3)
    >>> for train_data, test_data, train_data2, test_data2 in kfold.split(list1, list2):
    >>>    print(train_data, test_data, train_data2, test_data2)
    [2, 3] [1] [6, 7] [5]
    [1, 3] [2] [5, 7] [6]
    [1, 2] [3] [5, 6] [7]
    
    '''
    def __init__(self, n_splits, shuffle=False, random_state=None):
         super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            
    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

    def split(self, *args):
        """Generate splits of the data.

        Parameters
        ----------
        *args : Data or list-like objects
            Sets of data which will be split into train and test groups.

        Yields
        ------
        train : Data or list-like objects
            The training set for that split.
        test : Data or list-like objects
            The testing set for that split.
        """
        data = indexable(*args)
        n_samples = len(data[0])
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        for train, test in super().split(data[0]):
            tmp = list(
                    chain.from_iterable(
                        (_safe_indexing(a, train), _safe_indexing(a, test)) for a in data
                    )
                )
            for i, d in enumerate(data):
                if isinstance(d, Data):
                    tmp[2*i] = Data(tmp[2*i], strict=False)
                    tmp[2*i+1] = Data(tmp[2*i+1], strict=False)
            yield tmp
                    
    

