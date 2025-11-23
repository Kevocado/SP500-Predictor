import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .feature_engineering import create_features

def evaluate_model(model, df, ticker="SPY"):
    """
    Evaluates the model on the provided historical data.
    
    Args:
        model: Trained LightGBM model.
        df (pd.DataFrame): Historical data (OHLCV).
        ticker (str): Ticker symbol.
        
    Returns:
        pd.DataFrame: DataFrame with 'Actual', 'Predicted', 'Error', 'Rolling_MAE'.
        dict: Overall metrics {'MAE': float, 'RMSE': float}.
    """
    # Create features
    # We need to drop NaNs to get valid rows for prediction, but we want to keep the index
    df_features = create_features(df)
    
    # The target is 'target_next_hour' which is Close shifted by -60.
    # So at time T, we have features(T) and we predict Close(T+60).
    # We want to compare Prediction(T) vs Actual(T+60).
    
    # Drop rows where we don't have a target (the last 60 mins) for evaluation
    df_eval = df_features.dropna(subset=['target_next_hour'])
    
    if df_eval.empty:
        return pd.DataFrame(), {}
    
    # Load feature names to ensure correct columns
    import joblib
    import os
    # Assuming this is running from src/ or similar, we need to find the model dir
    # But we can just use the columns from the dataframe if we filter correctly
    # Better to rely on the model object if possible, or use the same logic as model.py
    # Let's try to predict using the columns present in df_eval that match numeric types
    # A safer way is to use the same feature selection logic as training
    
    target_col = 'target_next_hour'
    drop_cols = [target_col, 'cum_vol', 'cum_vol_price']
    feature_cols = [c for c in df_eval.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_eval[c])]
    
    X = df_eval[feature_cols]
    y_actual = df_eval[target_col]
    
    # Predict
    y_pred = model.predict(X)
    
    # Create result DataFrame
    results = pd.DataFrame(index=df_eval.index)
    results['Actual'] = y_actual
    results['Predicted'] = y_pred
    results['Error'] = results['Actual'] - results['Predicted']
    results['Abs_Error'] = results['Error'].abs()
    
    # Calculate Rolling Accuracy (e.g., 60-minute rolling MAE)
    results['Rolling_MAE'] = results['Abs_Error'].rolling(window=60).mean()
    
    # Overall Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    metrics = {'MAE': mae, 'RMSE': rmse}
    
    return results, metrics
