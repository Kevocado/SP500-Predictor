import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_data(ticker="SPY", period="1mo", interval="1m"):
    """
    Fetches intraday data for the given ticker.
    
    Args:
        ticker (str): Symbol to fetch (default "SPY").
        period (str): Data period to download (default "1mo").
        interval (str): Data interval (default "1m").
        
    Returns:
        pd.DataFrame: DataFrame with Datetime index and OHLCV columns.
    """
    # Map friendly names to tickers
    ticker_map = {
        "SPX": "^GSPC",
        "Nasdaq": "^NDX",
        "SPY": "SPY"
    }
    symbol = ticker_map.get(ticker, ticker)
    
    print(f"Fetching data for {symbol}...")
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if data.empty:
        print("No data found.")
        return pd.DataFrame()
    
    # Flatten MultiIndex columns if present (yfinance update)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    # Ensure Datetime index is timezone-aware (usually is, but good to check)
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        data.index = data.index.tz_convert('US/Eastern')
        
    data = data.dropna()
    print(f"Fetched {len(data)} rows.")
    return data

def save_data(df, filepath="Data/spy_data.csv"):
    """Saves data to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")

def load_data(filepath="Data/spy_data.csv"):
    """Loads data from CSV."""
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return None
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

if __name__ == "__main__":
    # Test fetching
    df = fetch_data()
    if not df.empty:
        save_data(df)
        print(df.head())
        print(df.tail())
