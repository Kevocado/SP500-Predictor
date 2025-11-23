#!/usr/bin/env python3
"""Test script to verify auto-retraining functionality."""

import sys
sys.path.insert(0, 'src')

from data_loader import fetch_data
from feature_engineering import create_features
from model import load_model, predict_next_hour, FeatureMismatchError

# Test with BTC (model file was deleted)
ticker = "BTC"

print(f"\n{'='*60}")
print(f"Testing Auto-Retrain for {ticker}")
print(f"{'='*60}\n")

# Step 1: Try to load model
print("Step 1: Loading model...")
model, needs_retrain = load_model(ticker=ticker)

if needs_retrain:
    print(f"✅ Correctly detected that model needs retraining!")
    print(f"   Model exists: {model is not None}")
    print(f"   Needs retrain flag: {needs_retrain}")
else:
    print(f"❌ Model should need retraining but doesn't!")

# Step 2: Fetch data and create features
print("\nStep 2: Fetching data and creating features...")
df = fetch_data(ticker=ticker, period="5d", interval="1m")
print(f"   Fetched {len(df)} rows of data")

df_features = create_features(df)
print(f"   Created features: {df_features.shape[1]} columns")
print(f"   Feature columns: {list(df_features.columns[:10])}...")

# Step 3: Try prediction (should fail or trigger retrain in app)
if model:
    print("\nStep 3: Attempting prediction...")
    try:
        pred = predict_next_hour(model, df_features, ticker=ticker)
        print(f"   Prediction: ${pred:,.2f}")
    except FeatureMismatchError as e:
        print(f"   ✅ FeatureMismatchError caught as expected!")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
else:
    print("\nStep 3: Skipping prediction (model is None)")

print(f"\n{'='*60}")
print("Test complete! The app should auto-retrain when you run it.")
print(f"{'='*60}\n")
