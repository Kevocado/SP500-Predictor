import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import dateutil.relativedelta
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv()

# Alpaca API Credentials (loaded from ENV)
API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

def get_macro_data():
    """Fetches VIX and 10Y-2Y Yield Curve data from FRED."""
    from fredapi import Fred
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        return {"vix": 20, "yield_curve": 0} # Fallback
    
    fred = Fred(api_key=fred_key)
    try:
        # VIX Index
        vix = fred.get_series('VIXCLS').iloc[-1]
        # 10-Year vs 2-Year Treasury Yield Spread (Recession Indicator)
        yc = fred.get_series('T10Y2Y').iloc[-1]
        return {"vix": vix, "yield_curve": yc}
    except Exception:
        return {"vix": 20, "yield_curve": 0}

def fetch_data(ticker="SPY", period="1mo", interval="1m"):
    """
    Fetches intraday data for the given ticker using Alpaca.

    Args:
        ticker (str): Symbol to fetch (default "SPY").
        period (str): Data period to download (default "1mo").
                      Supports '1d', '5d', '1mo', '3mo', '6mo', '1y'.
        interval (str): Data interval (default "1m").
                      Supports '1m', '5m', '15m', '1h', '1d'.

    Returns:
        pd.DataFrame: DataFrame with Datetime index and OHLCV columns.
    """
    # Map friendly names to Alpaca tickers
    ticker_map = {
        "SPX": "SPY",  # Alpaca doesn't support SPX index directly, use SPY proxy
        "Nasdaq": "QQQ", # Alpaca doesn't support NDX directly, use QQQ proxy
        "SPY": "SPY",
        "NVDA": "NVDA",
        "TSLA": "TSLA",
        "BTC": "BTC/USD",
        "ETH": "ETH/USD"
    }

    symbol = ticker_map.get(ticker, ticker)
    is_crypto = "/" in symbol

    print(f"Fetching data for {symbol} via Alpaca...")

    # Parse period to calculate start date
    now = datetime.now(timezone.utc)
    if period == "1d":
        start_date = now - timedelta(days=1)
    elif period == "5d":
        start_date = now - timedelta(days=5)
    elif period == "1mo":
        start_date = now - dateutil.relativedelta.relativedelta(months=1)
    elif period == "3mo":
        start_date = now - dateutil.relativedelta.relativedelta(months=3)
    elif period == "6mo":
        start_date = now - dateutil.relativedelta.relativedelta(months=6)
    elif period == "1y":
        start_date = now - dateutil.relativedelta.relativedelta(years=1)
    else:
        start_date = now - dateutil.relativedelta.relativedelta(months=1) # default

    # Parse interval to Alpaca TimeFrame
    if interval == "1m":
        timeframe = TimeFrame.Minute
    elif interval == "5m":
        timeframe = TimeFrame.Minute * 5
    elif interval == "15m":
        timeframe = TimeFrame.Minute * 15
    elif interval == "1h":
        timeframe = TimeFrame.Hour
    elif interval == "1d":
        timeframe = TimeFrame.Day
    else:
        timeframe = TimeFrame.Minute # default

    try:
        if not API_KEY or not SECRET_KEY:
            print(f"⚠️ Missing Alpaca API credentials. Cannot fetch {symbol}.")
            return pd.DataFrame()

        if is_crypto:
            crypto_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date
            )
            bars = crypto_client.get_crypto_bars(request_params)
        else:
            stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date
            )
            bars = stock_client.get_stock_bars(request_params)

        if not bars or not bars.data or symbol not in bars.data:
            print("No data found.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = bars.df

        # Alpaca returns a MultiIndex (symbol, timestamp). Drop symbol level.
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        # Ensure correct column names
        # Alpaca returns 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Keep only the required OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Ensure Datetime index is timezone-aware (Alpaca returns UTC by default)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        else:
            df.index = df.index.tz_convert('US/Eastern')

        df = df.dropna()
        print(f"Fetched {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error fetching data from Alpaca: {e}")
        return pd.DataFrame()

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
