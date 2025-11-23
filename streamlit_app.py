import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import fetch_data
from src.feature_engineering import create_features, prepare_training_data
from src.model import load_model, predict_next_hour, train_model

st.set_page_config(page_title="SP500 Hourly Predictor", layout="wide")

st.title("📈 S&P 500 Hourly Predictor")
st.markdown("Predicting the next hour's closing price using intraday data.")

# Sidebar for controls
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Select Index", ["SPX", "Nasdaq", "SPY"])

if st.sidebar.button("Retrain Model"):
    with st.status(f"Retraining model for {selected_ticker}...", expanded=True) as status:
        st.write(f"Fetching data for {selected_ticker}...")
        df = fetch_data(ticker=selected_ticker, period="7d", interval="1m")
        
        if not df.empty:
            st.write(f"Data fetched: {len(df)} rows. Processing features...")
            df_processed = prepare_training_data(df)
            
            st.write("Training LightGBM model...")
            train_model(df_processed, ticker=selected_ticker)
            
            status.update(label="Model retrained successfully!", state="complete", expanded=False)
            st.sidebar.success("Model retrained!")
        else:
            status.update(label="Failed to fetch data.", state="error")
            st.sidebar.error("Failed to fetch data.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Market Data ({selected_ticker})")
    # Fetch latest data
    with st.spinner("Loading data..."):
        df = fetch_data(ticker=selected_ticker, period="5d", interval="1m")
        
    if not df.empty:
        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name=selected_ticker))
        fig.update_layout(title=f"{selected_ticker} Intraday Price (Last 5 Days)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not load market data.")

with col2:
    st.subheader("Prediction")
    
    model = load_model(ticker=selected_ticker)
    
    if model is None:
        st.warning(f"Model for {selected_ticker} not found. Please train the model using the sidebar button.")
    elif not df.empty:
        # Prepare features for inference
        # We need to pass the dataframe to create_features to get the lag values
        # We don't dropna here because we want the last row even if it has NaNs for target
        df_features = create_features(df)
        
        # Get prediction
        try:
            prediction = predict_next_hour(model, df_features, ticker=selected_ticker)
            current_price = df['Close'].iloc[-1]
            
            st.metric(label="Current Price", value=f"${current_price:.2f}")
            st.metric(label="Predicted Next Hour Close", value=f"${prediction:.2f}", delta=f"{prediction - current_price:.2f}")
            
            st.info(f"Prediction Time: {df.index[-1]}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

st.markdown("---")
st.markdown("### Model Info")
if model:
    st.write(f"Model Type: LightGBM Regressor")
    # Could add feature importance plot here
