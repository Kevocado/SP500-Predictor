import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_DIR = "model"

def get_model_path(ticker):
    return os.path.join(MODEL_DIR, f"lgbm_model_{ticker}.pkl")

def train_model(df, ticker="SPY"):
    """
    Trains a LightGBM model to predict next hour close.
    
    Args:
        df (pd.DataFrame): Dataframe with features and target 'target_next_hour'.
        ticker (str): Ticker name to save model for.
    """
    save_path = get_model_path(ticker)
    # Define features and target
    target_col = 'target_next_hour'
    drop_cols = [target_col, 'cum_vol', 'cum_vol_price'] # Drop intermediate calc cols if any
    
    # Filter only numeric columns for features
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Training on {len(X)} samples with {len(feature_cols)} features.")
    
    # Time Series Split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    # Simple training on the last split for final model, or retrain on all?
    # Let's do a validation loop to print metrics, then train on all data.
    
    fold = 1
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data], 
                        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)])
        
        preds = bst.predict(X_val, num_iteration=bst.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"Fold {fold} RMSE: {rmse:.4f}")
        fold += 1
        
    # Train on all data
    print("Retraining on full dataset...")
    full_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, full_train_data, num_boost_round=100)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(final_model, save_path)
    print(f"Model saved to {save_path}")
    
    # Save feature names to ensure consistency during inference
    joblib.dump(feature_cols, os.path.join(os.path.dirname(save_path), f"features_{ticker}.pkl"))
    
    return final_model

def load_model(ticker="SPY"):
    """Loads the trained model for the given ticker."""
    model_path = get_model_path(ticker)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
    return joblib.load(model_path)

def predict_next_hour(model, current_data_df, ticker="SPY"):
    """
    Predicts the next hour close given the latest data.
    
    Args:
        model: Trained LightGBM model.
        current_data_df (pd.DataFrame): Dataframe containing the latest data points (needs history for features).
        ticker (str): Ticker to load feature names for.
        
    Returns:
        float: Predicted price.
    """
    # We need to generate features for the last row
    # Assumes current_data_df has enough history for lag features
    
    # Load feature names
    feature_names_path = os.path.join(MODEL_DIR, f"features_{ticker}.pkl")
    if os.path.exists(feature_names_path):
        feature_cols = joblib.load(feature_names_path)
    else:
        # Fallback if feature list missing (shouldn't happen if trained)
        raise FileNotFoundError(f"Feature list not found for {ticker}. Train model first.")

    # Get the last row of features
    last_row = current_data_df.iloc[[-1]][feature_cols]
    
    prediction = model.predict(last_row)
    return prediction[0]
