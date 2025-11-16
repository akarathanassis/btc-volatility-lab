"""
Data loader for BTC OHLC data using Yahoo Finance.

This module currently provides:
- load_btc_ohlc: Downloads BTC-USD OHLCV data via yfinance
                  Saves locally under /data/
                  Returns a pandas DataFrame
"""

from pathlib import Path
import pandas as pd
import yfinance as yf


# Locate project-level /data directory (two levels up from this file)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)  # ensure directory exists


def load_btc_ohlc(period: str = "2y", interval: str = "1h", save: bool = True) -> pd.DataFrame:
    """
    Load BTC OHLCV data from Yahoo Finance (BTC-USD ticker).

    Parameters
    ----------
    period : str
        Lookback period (e.g., '1y', '2y', 'max').
    interval : str
        Candle interval (e.g., '1h', '1d', '15m').
    save : bool
        Whether to save the downloaded dataframe to /data/.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by timestamp with:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """

    print(f"Downloading BTC-USD data from Yahoo Finance: period={period}, interval={interval}")

    df = yf.download(
        "BTC-USD",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    # Ensure datetime index & drop empty rows
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    # Save to file
    if save:
        out_path = DATA_DIR / f"btc_ohlc_{period}_{interval}.parquet"
        df.to_parquet(out_path)
        print(f"Saved data to {out_path}")

    print(f"Loaded {len(df)} rows of BTC-USD price data")
    return df
