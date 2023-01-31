from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Union, Set
import pandas as pd



class Frequency(Enum):
  HOURLY = "Hourly"
  DAILY = "Daily"
  MONTHLY = "Monthly"

class Quote:
  def __init__(self, symbol: str, open: float, high: float, low: float, 
                close: float, adj_close: float, volume: int, ts: datetime, **kwargs) -> None:
      self.symbol = symbol
      self.open = open
      self.high = high
      self.low = low
      self.close = close
      self.adj_close = adj_close
      self.volume = volume
      self.ts = ts
      self.__dict__.update(kwargs)

  def __str__(self) -> str:
    return(f'Symbol : {self.symbol} | Close : {self.close} | ts : {self.ts}')


class Config:
  def __init__(self, universe: List[str], strategy_code: str, frequency: Frequency, model_parameters: dict, 
                 quote_list: List[Quote], start_ts_backtest: datetime, end_ts_backtest: datetime) -> None:
    self.universe = universe
    self.strategy_code = strategy_code
    self.frequency = frequency
    self.model_parameters = model_parameters
    self.calendar = []
    self.get_market_date(quote_list)
    self.start_ts_backtest = start_ts_backtest
    self.end_ts_backtest = end_ts_backtest
    
  def __post_init__(self):
    if self.start_ts >= self.end_ts:
        raise ValueError("self.start_ts must be before self.end_ts")
    if len(self.universe) == 0:
        raise ValueError("self.universe should contains at least one element")
            
  @property
  def timedelta(self):
    if self.frequency == Frequency.HOURLY:
        return timedelta(hours=1)
    if self.frequency == Frequency.DAILY:
        return timedelta(days=1)
            
  def get_market_date(self, quote_list: List[Quote]) -> None:
    """Fill a list of market dates (datetime) for the period considered in the study"""
    for quote in quote_list:
      if len(self.calendar) == 0:
        self.calendar.append(quote.ts)
      elif self.calendar[-1] != quote.ts:
        self.calendar.append(quote.ts)

  def get_last_market_date(self, ts: datetime):
    """Return the last market date"""
    date = ts - self.timedelta
    res = None
    while res is None and date >= self.start_ts_backtest:
      if date in self.calendar:
          res = date
          break
      date = date - self.timedelta
    return res        

class PositionType(Enum):
  LONG = 1
  SHORT = 0

class Position:
  def __init__(self, underlying_code: str, ts: datetime, value: PositionType):
    """
    This object is a position for a given Symbol at a given time
    """
    self.underlying_code = underlying_code
    self.ts = ts
    self.value = value
  
  def __str__(self) -> str:
    return f"Symbol : {self.underlying_code} | Position : {self.value} | ts : {self.ts}"

class Weight:
  """Store the weight for a symbol and a given strategy at a given time"""
  def __init__(self, product_code: str, underlying_code: str, ts: datetime, value: float) -> None:
    self.product_code = product_code
    self.underlying_code = underlying_code
    self.ts = ts
    self.value = value

class BaseWeightComputation:
  def __init__(self, strategy_code: str):
    self.strategy_code = strategy_code
    self._weight_by_pk = dict()

  def compute_weights(self, ts: datetime, data: List[Quote]) -> List[Weight]:
    raise NotImplementedError

  def get_weight(self, underlying_code: str,ts: datetime) -> Weight:
    return self._weight_by_pk.get((self.strategy_code,underlying_code,ts))
    
  @property
  def weight_by_pk(self) -> dict:
    return self._weight_by_pk

class EquallyWeightedStrategy(BaseWeightComputation):
  @staticmethod
  def count_nb_asset_in_ptf(pos_list: List[Position]) -> int:
    """
    Parameters
    ----------
    pos_list : List[Position]
               List of position at a given time for different symbols
    Returns
    -------
    count : int
            Returns the number of assets that we hold in the ptf at a given time
    """
    count = 0
    for position in pos_list:
        if position.value == PositionType.LONG:
            count += 1
    return count

  def compute_weights(self, pos_list: List[Position], ts: datetime) -> List[Weight]:
    """
    Parameters
    ----------
    pos_list : List[Position]
               List of position at a given time for different symbols
    ts : datetime
         Date at which we want to compute weight
    Returns
    -------
    weight_list : List[Weight]
                  Returns the list of weight for the considered symbols at this date
    """
    nb_assets_ptf = EquallyWeightedStrategy.count_nb_asset_in_ptf(pos_list)
    weight_list = []
    for position in pos_list:
      if position.value == PositionType.LONG:
        value = 1/nb_assets_ptf
      else:
        value = 0
      w_tmp = Weight(
          product_code=self.strategy_code,
          underlying_code=position.underlying_code,
          ts=position.ts,
          value=value
      )
      key = (self.strategy_code, position.underlying_code, position.ts)
      self._weight_by_pk[key] = w_tmp
      weight_list.append(w_tmp)
    return weight_list

