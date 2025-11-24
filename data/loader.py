"""
Data loader for BTC OHLC data using Yahoo Finance.

This module currently provides:
- load_btc_ohlc: Downloads BTC-USD OHLCV data via yfinance
                  Saves locally under /data/
                  Returns a pandas DataFrame
"""

from pathlib import Path
import pandas as pd
import requests
import glob
import os


BASE = "https://api.binance.com"
endpoint = "/api/v3/klines"


# Locate project-level /data directory (two levels up from this file)
DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(exist_ok=True)  # ensure directory exists


def get_klines(symbol="BTCUSDT", interval="1h", limit=1000):
    print("Downloading klines data from Binance...")
    url = BASE + endpoint
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params).json()

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades",
        "tbbv","tbqv","ignore"
    ]

    df = pd.DataFrame(data, columns=cols)

    # convert numeric columns
    num_cols = cols[1:-1]
    df[num_cols] = df[num_cols].apply(pd.to_numeric)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df
