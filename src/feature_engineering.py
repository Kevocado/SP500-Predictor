import pandas as pd
import numpy as np
import ta

def create_features(df):
    """
    Generates technical indicators and time-based features.
    
    Args:
        df (pd.DataFrame): Dataframe with OHLCV data.
        
    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    df = df.copy()
    
    # Ensure we have the necessary columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Time Features ---
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    
    # --- Technical Indicators ---
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['Close']
    
    # VWAP (Intraday) - Approximate using cumulative sum since start of day
    # Note: A true VWAP resets daily. Here we can do a rolling calculation or group by day.
    # For simplicity in this rolling window context, we'll use a rolling VWAP or just Volume/Price features.
    # Let's implement a daily reset VWAP.
    df['cum_vol'] = df.groupby(df.index.date)['Volume'].cumsum()
    df['cum_vol_price'] = df.groupby(df.index.date).apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['dist_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
    
    # Returns and Volatility
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility_30m'] = df['log_ret'].rolling(window=30).std()
    
    # Lag Features
    for lag in [1, 5, 15, 30, 60]:
        df[f'lag_close_{lag}'] = df['Close'].shift(lag)
        df[f'lag_ret_{lag}'] = df['Close'].pct_change(periods=lag)
        
    # --- Target Creation ---
    # Predict next hour close (60 minutes ahead)
    # We want to predict Close(t+60).
    # So Target(t) = Close(t+60)
    df['target_next_hour'] = df['Close'].shift(-60)
    
    # Drop NaN values created by lags and indicators (but keep the end for inference if needed, 
    # though for training we must drop target NaNs)
    # We will handle dropping in the training function.
    
    return df

def prepare_training_data(df):
    """Prepares data for training by dropping NaNs."""
    df = create_features(df)
    df = df.dropna()
    return df
