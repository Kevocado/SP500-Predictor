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
    
    # Calculate Directional Accuracy
    # Did the model correctly predict if price would go up or down relative to the price at prediction time?
    # We need the price at time T (when prediction was made).
    # Since we shifted target by -60, the price at index T is the "current" price at time T.
    results['Price_At_Pred'] = df_eval['Close']
    results['Actual_Dir'] = np.sign(results['Actual'] - results['Price_At_Pred'])
    results['Pred_Dir'] = np.sign(results['Predicted'] - results['Price_At_Pred'])
    results['Correct_Dir'] = (results['Actual_Dir'] == results['Pred_Dir']).astype(int)
    
    # Overall Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    accuracy = results['Correct_Dir'].mean()
    
    # --- Trust Metrics (Brier, Calibration, PnL) ---
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    
    # 1. Brier Score (Probabilistic Error)
    # We need probabilities for Direction (Up).
    # Simple proxy: If Pred > Current, Prob(Up) = 1.0 (Binary). 
    # For a regressor, we don't have native probs unless we use the Z-score method.
    # Let's use the Z-score method we use in the app.
    # Z = (Pred - Current) / RMSE
    # Prob(Up) = CDF(Z)
    import scipy.stats as stats
    
    # Calculate RMSE dynamically or use the overall RMSE
    # Using overall RMSE for simplicity
    z_scores = (results['Predicted'] - results['Price_At_Pred']) / rmse
    probs_up = stats.norm.cdf(z_scores)
    
    # Actual Outcome (1 if Up, 0 if Down)
    actual_outcomes = (results['Actual'] > results['Price_At_Pred']).astype(int)
    
    brier = brier_score_loss(actual_outcomes, probs_up)
    
    # 2. Calibration Curve
    prob_true, prob_pred = calibration_curve(actual_outcomes, probs_up, n_bins=10)
    
    # 3. PnL Backtest (Simple Strategy)
    # Strategy: Bet $100 on "Yes" if Prob > 60%, Bet $100 on "No" if Prob < 40%.
    # Payout: $100 profit if correct, -$100 loss if wrong. (Simplified binary option)
    results['Bet'] = 0
    results['Bet'] = np.where(probs_up > 0.60, 1, results['Bet']) # Bet Up
    results['Bet'] = np.where(probs_up < 0.40, -1, results['Bet']) # Bet Down
    
    results['Outcome'] = 0
    # If Bet Up (1) and Actual Up (1) -> Win
    # If Bet Down (-1) and Actual Down (0) -> Win
    # Else Loss
    
    # Map Actual Down to -1 for comparison
    actual_signed = np.where(actual_outcomes == 1, 1, -1)
    
    results['PnL'] = np.where(results['Bet'] == 0, 0, 
                              np.where(results['Bet'] == actual_signed, 100, -100))
    
    results['Cum_PnL'] = results['PnL'].cumsum()
    
    metrics = {
        'MAE': mae, 
        'RMSE': rmse, 
        'Directional_Accuracy': accuracy, 
        'Correct_Count': results['Correct_Dir'].sum(), 
        'Total_Count': len(results),
        'Brier_Score': brier,
        'Calibration_Data': {'prob_true': prob_true, 'prob_pred': prob_pred},
        'Total_PnL': results['PnL'].sum()
    }
    
    # Calculate Daily Metrics
    daily_metrics = results.groupby(results.index.date).apply(
        lambda x: pd.Series({
            'MAE': x['Abs_Error'].mean(),
            'Accuracy': (x['Actual_Dir'] == x['Pred_Dir']).mean(),
            'Correct': (x['Actual_Dir'] == x['Pred_Dir']).sum(),
            'Total': len(x),
            'Daily_PnL': x['PnL'].sum()
        })
    )
    
    return results, metrics, daily_metrics
