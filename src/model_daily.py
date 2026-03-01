"""
Model Daily — LightGBM direction predictor for SPY/QQQ

Uses the full 20-feature pipeline from feature_engineering.py.
Targets: next-hour close price.
Sizing: Quarter-Kelly (0.25×).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from dotenv import load_dotenv

load_dotenv()


def prepare_daily_data(df, ticker="SPY"):
    """
    Prepares data using the full 3-cluster feature pipeline.
    Target: Close price at 16:00 same day.
    """
    from src.feature_engineering import create_features, FEATURE_COLUMNS

    df, gex_data = create_features(df, ticker)

    # Target: last close of day (EOD at 16:00)
    daily_closes = df['Close'].resample('D').last()
    df['Target_Close'] = df.index.normalize().map(daily_closes)

    # Drop NaN
    df = df.dropna(subset=FEATURE_COLUMNS + ['Target_Close'])

    return df, FEATURE_COLUMNS, gex_data


def train_daily_model(df, ticker="SPY"):
    """
    Trains LightGBM on the full 20-feature set.
    """
    df_proc, feature_cols, gex_data = prepare_daily_data(df, ticker)

    if len(df_proc) < 50:
        print(f"  ⚠️ Insufficient data ({len(df_proc)} rows). Need 50+ for training.")
        return None

    X = df_proc[feature_cols]
    y = df_proc["Target_Close"]

    # Time-based train/test split
    train_size = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # Calculate test RMSE
    preds = model.predict(X_test)
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    print(f"  📊 {ticker} Model RMSE: ${rmse:.2f}")

    # Save
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, f"model/lgbm_model_{ticker}.pkl")
    joblib.dump(feature_cols, f"model/features_{ticker}.pkl")
    print(f"  💾 Model saved: model/lgbm_model_{ticker}.pkl")

    return model


def predict_daily_close(model, current_df_features, feature_cols=None):
    """
    Predicts the Daily Close price using the trained model.
    """
    from src.feature_engineering import FEATURE_COLUMNS
    cols = feature_cols or FEATURE_COLUMNS

    X = current_df_features.reindex(columns=cols, fill_value=0)
    prediction = model.predict(X)[-1] if len(X) > 0 else None
    return prediction


def load_daily_model(ticker="SPY"):
    """
    Loads model from HuggingFace Hub, fallback to local.
    """
    local_model = f"model/lgbm_model_{ticker}.pkl"
    local_features = f"model/features_{ticker}.pkl"

    # Try HF Hub first
    try:
        from huggingface_hub import hf_hub_download
        repo_id = "Kevocado/sp500-predictor-models"
        cached = hf_hub_download(repo_id=repo_id, filename=f"lgbm_model_{ticker}.pkl",
                                 cache_dir="model", force_filename=f"lgbm_model_{ticker}.pkl")
        model = joblib.load(cached)
        print(f"  ✅ Loaded {ticker} model from HF Hub")
        return model
    except Exception as e:
        print(f"  ⚠️ HF Hub failed for {ticker}: {e}")

    # Local fallback
    if os.path.exists(local_model):
        print(f"  ✅ Loaded local {ticker} model")
        return joblib.load(local_model)

    print(f"  ❌ No model found for {ticker}")
    return None


def quarter_kelly(edge, prob, max_kelly_pct=6):
    """
    Quarter-Kelly position sizing.

    Full Kelly: f* = (edge × prob) / (1 - prob)
    Quarter-Kelly: f = f* × 0.25

    Capped at max_kelly_pct of bankroll.

    Args:
        edge: float, model edge (0.0 to 1.0)
        prob: float, model probability (0.0 to 1.0)
        max_kelly_pct: float, max % of bankroll per trade

    Returns:
        float: recommended position size as % of bankroll
    """
    if prob >= 1.0 or prob <= 0.0 or edge <= 0:
        return 0.0

    full_kelly = (edge * prob) / (1 - prob) * 100
    quarter = full_kelly * 0.25
    return min(quarter, max_kelly_pct)


if __name__ == "__main__":
    print("Testing Model Pipeline...")

    # Test Kelly sizing
    print("\nQuarter-Kelly Examples:")
    for edge, prob in [(0.25, 0.75), (0.15, 0.65), (0.10, 0.55)]:
        k = quarter_kelly(edge, prob)
        print(f"  Edge={edge:.0%}, Prob={prob:.0%} → Size={k:.1f}% of bankroll")

    # Test model loading
    print("\nModel Loading:")
    for t in ["SPY", "QQQ"]:
        m = load_daily_model(t)
        if m:
            print(f"  {t}: {type(m).__name__}")
