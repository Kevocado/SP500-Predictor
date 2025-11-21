import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import fetch_data
from feature_engineering import prepare_training_data, create_features
from model import train_model, predict_next_hour

def test_pipeline():
    print("Testing Data Loader...")
    df = fetch_data(period="5d", interval="1m")
    assert not df.empty, "Data fetch returned empty DataFrame"
    print("Data Loader OK.")
    
    print("Testing Feature Engineering...")
    df_processed = prepare_training_data(df)
    assert 'target_next_hour' in df_processed.columns, "Target column missing"
    assert 'rsi' in df_processed.columns, "RSI missing"
    print("Feature Engineering OK.")
    
    print("Testing Model Training...")
    model = train_model(df_processed, save_path="model/test_model.pkl")
    assert model is not None, "Model training failed"
    print("Model Training OK.")
    
    print("Testing Prediction...")
    # Create features for the original df (including last row)
    df_features = create_features(df)
    prediction = predict_next_hour(model, df_features)
    print(f"Prediction: {prediction}")
    assert isinstance(prediction, float), "Prediction is not a float"
    print("Prediction OK.")
    
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_pipeline()
