from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from Classes import *


"""Work with a copy of the DataFrame to avoid changing the original DataFrame."""

def standardize_feat_basket(data: pd.DataFrame, features_names: List[str], by: List[str]) -> pd.DataFrame:
    """
    Standardize a given list of features in a DataFrame with respect to the other stocks
    """
    mean = data.groupby(by)[features_names].transform('mean')
    sigma = data.groupby(by)[features_names].transform('std')
    res = data.copy()
    res[features_names] = (res[features_names] - mean) / sigma
    return res
    

def standardize_feat_regular(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a given list of features in a DataFrame in a regular way (x - mean)/std on the entire dataset
    """
    mu = data.mean()
    sig = data.std()
    res = data.copy()
    res = (res - mu)/sig
    return res
        

def add_return (data: pd.DataFrame, target: str, deltas: List[int], 
                by: str = "symbol", disable_tqdm: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with Multindex (Date, Symbol) and Open, High, Low, Close, Adj Close, Volume features
    target : str
        Feature for which we want to compute returns
    deltas : List[int]
        List of Lags for which we compute the returns
    by : str
        Feature linked to the groupby
    Returns
    -------
    res : pd.DataFrame
        Initial df with additionnal return features 
    """
    res = data.copy()
    # Compute the returns
    for delta in tqdm(deltas, desc="Computing returns", disable=disable_tqdm):
        # Past return
        if delta > 0:
            feature_name = 'past_ret_' + str(delta) + 'D_' + target
            res[feature_name] = res.groupby(by)[target].pct_change(periods=delta, limit=0)
        # Future return
        else:
            feature_name = 'fut_ret_' + str(abs(delta)) + 'D_' + target
            res[feature_name] = res.groupby(by)[target].pct_change(periods=abs(delta), limit=0)
            res[feature_name] = res.groupby(by)[feature_name].shift(-abs(delta))
    return res

def binarize_label(data: pd.Series) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with Multindex (Date, Symbol) and some features
    Returns
    -------
    data : pd.DataFrame
        Initial DataFrame with the binary feature added
    """
    data = data.groupby("ts").apply(lambda df: pd.Series((df > df.median()).astype(int), index=df.index))
    return data



    
    



