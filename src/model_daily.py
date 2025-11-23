import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import time
import joblib
import os
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def prepare_daily_data(df):
    """
    Prepares data for Daily Close prediction.
    Target: The Close price at 16:00 (4 PM) of the SAME day.
    """
    df = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 1. Base Features (Technical Indicators)
    # We use the same indicators but on hourly data
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()
    
    sma = SMAIndicator(close=df["Close"], window=20)
    df["SMA_20"] = sma.sma_indicator()
    
    df["Dist_from_SMA"] = df["Close"] - df["SMA_20"]
    
    # 2. Time Features (Crucial for EOD prediction)
    # "How much time is left in the day?"
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    
    # 3. Daily Context
    # We need the "Open" price of the current day to know how much we've moved
    # Since we are using hourly data, we can group by Date and get the first 'Open'
    # But for a robust realtime calculation, we can just use the current row's data if it's intraday.
    # A simple approximation for "Daily Open" in a continuous dataframe:
    # We can resample to daily, get open, and reindex.
    
    daily_opens = df['Open'].resample('D').first()
    # Map back to hourly index
    # Forward fill daily opens to match the hourly timestamps
    df['Daily_Open'] = df.index.normalize().map(daily_opens)
    
    df['Dist_from_Open'] = df['Close'] - df['Daily_Open']
    df['Pct_Change_Day'] = (df['Close'] - df['Daily_Open']) / df['Daily_Open']
    
    # 4. Create Target: Close Price at 16:00 of the SAME day
    # We need to look ahead to find the 16:00 candle for this specific date.
    # This is tricky in a vectorized way.
    
    # Strategy:
    # 1. Get all 16:00 rows.
    # 2. Create a Series mapping Date -> 16:00 Close.
    # 3. Map this target back to every row in that day.
    
    # Filter for market close candles (15:00 or 16:00 depending on data source, usually 15:30-16:00 is the last one)
    # For hourly data, usually the 15:00 candle covers 15:00-16:00, or 16:00 is the closing print.
    # Let's assume 16:00 is the close. If not present, take the last available for that day.
    
    daily_closes = df['Close'].resample('D').last()
    df['Target_Close'] = df.index.normalize().map(daily_closes)
    
    # Drop NaN (e.g. if we don't have the close for today yet, we can't train on it)
    df = df.dropna()
    
    feature_cols = [
        "Close", "RSI", "Dist_from_SMA", 
        "Hour", "DayOfWeek", 
        "Dist_from_Open", "Pct_Change_Day"
    ]
    
    return df, feature_cols

def train_daily_model(df, ticker="SPX"):
    """
    Trains a LightGBM model to predict the Daily Close.
    """
    df_processed, feature_cols = prepare_daily_data(df)
    
    X = df_processed[feature_cols]
    y = df_processed["Target_Close"]
    
    # Train/Test Split (Time-based)
    train_size = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    
    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, f"model/lgbm_model_daily_{ticker}.pkl")
    print(f"Daily Model saved for {ticker}")
    
    return model

def predict_daily_close(model, current_df_features):
    """
    Predicts the Daily Close price.
    """
    # We expect current_df_features to be a single row or dataframe prepared by prepare_daily_data
    # Note: prepare_daily_data returns (df, cols). We just need the cols.
    
    # If input is just raw dataframe, we need to process it to get features
    # But usually for inference we pass the already processed features.
    # Let's assume we pass the processed row.
    
    feature_cols = [
        "Close", "RSI", "Dist_from_SMA", 
        "Hour", "DayOfWeek", 
        "Dist_from_Open", "Pct_Change_Day"
    ]
    
    # Ensure features match: reindex to align with model's expected features
    # Missing features will be filled with 0
    X = current_df_features.reindex(columns=feature_cols, fill_value=0)
    prediction = model.predict(X)[0]
    return prediction

def load_daily_model(ticker="SPX"):
    path = f"model/lgbm_model_daily_{ticker}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None
