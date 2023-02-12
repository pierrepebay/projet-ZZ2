from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Set, Tuple
from LSTM_model import LongShortTermMemory
from Classes import Quote, Config, Position, PositionType, Weight, EquallyWeightedStrategy
import matplotlib.pyplot as plt


class PositionGenerator:
    """Generate a list of positions at each trading date according to the ML model predictions"""
    def __init__(self, quotes_by: Dict[Tuple[str, datetime], Quote], ml_predictions: Dict[str, Dict[datetime, int]], config: Config):
      """
      Parameters
      ----------
      quotes_by: Dict[Tuple[str, datetime], Quote]
        Dictionary containing for each symbol a quote at a given time
      ml_predictions : Dictionary containing for each symbol a prediction at a given time
      config : config
        Contain necessary variables and parameters to perform the backtesting part
      """      
      self.quotes_by = quotes_by
      self.config = config
      self.ml_predictions = ml_predictions

    def get_previous_quotes(self, underlying_code: str, ts: datetime, n_steps: int) -> List[Quote]:
      """
      Returns 'n_steps' previous quotes for a given underlying
      Parameters
      ----------
      underlying_code : str
        name of the underlying asset
      ts : datetime
        Date below which we want to obtain previous quotes (t - 1, t - 2, etc)
      n_steps : int
        Length of the quote list to return
      Returns
      -------
      quote_list : List[Quote]
        List of previous quotes
      """
      quote_list = []
      date = ts - self.config.timedelta
      while len(quote_list) != n_steps and date >= self.config.start_ts_backtest:
        # If it's a quotation day
        if date in self.config.calendar:
          quote = self.quotes_by[(underlying_code, date)]
          quote_list.append(quote)
        date = date - self.config.timedelta
      return quote_list