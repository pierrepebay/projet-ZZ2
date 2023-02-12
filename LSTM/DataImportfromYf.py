import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from typing import List, Dict, Tuple
from Classes import Quote


class DataImportfromYf:
    def __init__(self, universe : List[str], start_ts : datetime, 
                 end_ts : datetime, interval : str, ignore_tz : bool, period : str = None) -> None:
        """
        Parameters
        ----------
        universe : List[str]
                List of all the tickers for which we want to get informations
        start_ts : datetime
                Beginning date
        end_ts :   datetime
                Ending date
        interval : str
                Fetch data by interval. Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        ignore_tz : bool
                Whether to ignore timezone when aligning ticker data from different timezones
        period :   str
                If we want to get data over a specific period and not from start/end date
                valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        """
        self.universe = universe
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.interval = interval
        self.ignore_tz = ignore_tz
        self.period = period


    def get_data(self) -> pd.DataFrame:
        """Get the data OHLC in a df format for all securities in the universe"""
        if self.period == None:
            self.data = yf.download(tickers = self.universe, start = self.start_ts, end = self.end_ts, 
                                    interval = self.interval,  ignore_tz = self.ignore_tz, group_by = 'tickers', threads = True)
        else:
            self.data = yf.download(tickers = self.universe, period = self.period, interval = self.interval, 
                                    ignore_tz = self.ignore_tz, group_by ='tickers', threads = True)
        # Creation of a Multindex (Date, Symbol) to simplify future treatments   
        self.data = self.data.stack(level=0)
        self.data.index = self.data.index.rename(["ts", "symbol"]) 
        self.data.sort_index(axis=0, level=["ts", "symbol"])
        self.data = self.data.rename(columns={"Date": "ts",
                                            "Open": "open",
                                            "High": "high",
                                            "Low": "low",
                                            "Adj Close": "adj_close",
                                            "Close": "close",
                                            "Volume": "volume"})  
        return self.data
    
    @staticmethod
    def data_to_quote_list(df : pd.DataFrame) -> List[Quote]:
        """
        Parameters
        ----------
        df : pd.DataFrame
                Dataframe with Multindex (Date, Symbol) and Open, High, Low, Close, Adj Close, Volume features
        Returns
        -------
        quote_list : List[Quote]
                Scan the dataframe and transform it into a list of quote objects
        """
        data = df.copy()  
        data = data.reset_index()         
        data_dict = data.to_dict(orient='index')
        quote_list = []
        for key, values in data_dict.items():
            quote = Quote(**values)
            quote_list.append(quote)
        return quote_list
    

