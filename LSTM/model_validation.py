import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple



def get_train_val_test_idx_rolling(data: pd.DataFrame, n_train: int, n_val: int, 
                         n_test: int, rolling: int = 30) -> Tuple[List[pd.MultiIndex], List[pd.MultiIndex], List[pd.MultiIndex]]:
    """
    This method is suited for time series and returns necessary indexes to perform model validation with a rolling window.
    Parameters
    ----------
    df : pd.DataFrame
        Data for one given symbol on a given period 
    n_train : int
        Number of samples to consider for each training block
    n_val : int
        Number of samples to consider for each validation block
    n_test : int
        Number of samples to consider for each test block
    rolling : int
        rolling window to consider for each cycle train-val-test
    Returns
    -------
    train_idx, val_idx, test_idx : Tuple[List[pd.MultiIndex], List[pd.MultiIndex], List[pd.MultiIndex]]
        Each list contains relevant indexes to consider for each cycle train-val-test 
    """
    train_idx = []
    val_idx = []
    test_idx = []
    
    # Length of the total chain train-val-test
    seq_length = n_train + n_val + n_test
    # Number of possible train-val-test cycles along the time series 'data'
    n_steps = int(np.ceil((len(data) - seq_length)/rolling))

    for i in range(n_steps):
        train_idx.append(data[(rolling * i):(n_train + rolling * i)].index)
        val_idx.append(data[(n_train + rolling * i):(n_val + n_train + rolling * i)].index)
        # Handle the case where we didn't have enough data to put rolling obs (30 by default) in the test set
        if i == n_steps - 1 and (seq_length + rolling * i) > len(data):
            test_idx.append(data[(n_val + n_train + rolling * i):].index)
        else:
            test_idx.append(data[(n_val + n_train + rolling * i):(seq_length + rolling * i)].index)

    return train_idx, val_idx, test_idx

def get_train_val_test_idx_regular(data: pd.DataFrame, ratio_train: float, 
                                   ratio_val: int) -> Tuple[List[pd.MultiIndex], List[pd.MultiIndex], List[pd.MultiIndex]]:
        """
        This method is suited for time series and returns necessary indexes to perform model validation in a regular way, i.e.
        we have only one train, val and test
        Parameters
        ----------
        df : pd.DataFrame
            Data for one given symbol on a given period 
        ratio_train : float
            ratio of training samples on the entire series
        ratio_val : float
            ratio of validation samples on the entire series
        Returns
        -------
        train_idx, val_idx, test_idx : Tuple[List[pd.MultiIndex], List[pd.MultiIndex], List[pd.MultiIndex]]
            Each list contains relevant indexes to consider for the split train-val-test 
        """
        assert 0 < (ratio_train + ratio_val) < 1, "The sum of ratio_train and ratio_val must be between 0 and 1 (bounds excluded)"
        train_idx = []
        val_idx = []
        test_idx = []

        n = data.shape[0]
        # Nb of samples to put in each set
        n_train = int(ratio_train * n)
        n_val = int(ratio_val * n)

        train_idx.append(data.iloc[:n_train].index)
        val_idx.append(data.iloc[(n_train):(n_train + n_val)].index)
        test_idx.append(data.iloc[(n_train + n_val):].index)

        return train_idx, val_idx, test_idx
        