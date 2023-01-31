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
      quotes_by : Dict[Tuple[str, datetime], Quote]
        Dictionary containing for each symbol a quote at a given time
      ml_predictions : Dictionary containing for each symbol a prediction at a given time
      config : Config
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
 
    def compute_positions(self, ts: datetime) -> List[Position]:
      """
      Parameters
      ----------
      ts : Date for which we want to get positions
      Returns
      -------
      pos_list : List[Position]
        Returns a list of Position for the symbols contained in the portfolio at this time
      """
      pos_list = []
      last_ts = self.config.get_last_market_date(ts=ts)
      for underlying_code in self.config.universe:
        if (underlying_code, ts) in self.quotes_by and (underlying_code, last_ts) in self.quotes_by:
          y_pred = self.ml_predictions[underlying_code][ts]
          if y_pred == 0:
            pos = Position(underlying_code=underlying_code, ts=ts, value=PositionType.SHORT)
          elif y_pred == 1:
            pos = Position(underlying_code=underlying_code, ts=ts, value=PositionType.LONG)
          pos_list.append(pos)
      return pos_list


class Backtester:
    """
    Perform the backtest of the strategy
    """
    def __init__(self, config: Config, quote_list: List[Quote], ml_predictions: Dict[str, Dict[datetime, int]]):
      self._config = config
      self._start_ts_backtest = config.start_ts_backtest
      self._end_ts_backtest = config.end_ts_backtest
      self._calendar = config.calendar
      self._universe = config.universe 
      self._timedelta = config.timedelta 
      self._strategy = EquallyWeightedStrategy(config.strategy_code)
      self._quote_by_pk = dict() 
      self._generate_quotes_dict(quote_list)
      self._pos_gen = PositionGenerator(self._quote_by_pk, ml_predictions, self._config)
      self.perfs_list = []

    def _generate_quotes_dict(self, quote_list: List[Quote]):
      """Generate a quote dictionary with (symbol, ts) keys"""
      for quote in quote_list:
        underlying_code = quote.symbol
        ts = quote.ts
        self._quote_by_pk[(underlying_code, ts)] = quote
    
    def _get_start_date_predict(self):
      """
      Returns the date from which we start to make predictions                               
      """
      n_steps = self._config.model_parameters["n_steps"]
      date = self._start_ts_backtest
      count = 0
      while count != n_steps + 1 and date <= self._end_ts_backtest:
        # If it's a quotation day
        if date in self._calendar:
            count += 1
        date = date + self._timedelta
      return date - self._timedelta
    
    def _compute_perf(self, ts: datetime):
      """Compute the performance of the portfolio at a given date"""
      perf_ = 0
      pos_list = self._pos_gen.compute_positions(ts)
      weights_list = self._strategy.compute_weights(pos_list, ts)
      for weight in weights_list:
        underlying_code = weight.underlying_code
        value = weight.value
        current_quote = self._quote_by_pk.get((underlying_code, ts))
        previous_quote =  self._pos_gen.get_previous_quotes(underlying_code=underlying_code, ts=ts, n_steps=1)[0]
        if current_quote is not None and previous_quote is not None:
          perf_ += value * (current_quote.close / previous_quote.close - 1)
        else:
          raise ValueError(f'missing quote for {underlying_code} at {ts} or the quote before')
      return perf_
    
    def run_backtest(self):
      """Main function of the backtester allowing us to compute the perf on the total trading period"""
      perfs_list = []
      tmp_date = self._get_start_date_predict()
      while tmp_date <= self._end_ts_backtest:
        # If it's a quotation day
        if tmp_date in self._calendar:
            perf = self._compute_perf(ts=tmp_date)
            perfs_list.append(perf)
        tmp_date = tmp_date + self._config.timedelta
      self.perfs_list = perfs_list
      return perfs_list
  


        