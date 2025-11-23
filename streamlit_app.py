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

from src.evaluation import evaluate_model
from src.utils import get_market_status
from datetime import timedelta

# ... (Previous code remains)

# Main content
st.title(f"📈 {selected_ticker} Hourly Predictor")

# Market Status Indicator
status = get_market_status()
st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {'#1b4d1b' if status['is_open'] else '#4d1b1b'}; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
        <span style="height: 12px; width: 12px; background-color: {status['color']}; border-radius: 50%; display: inline-block;"></span>
        <span style="font-weight: bold; color: white;">{status['status_text']}</span>
        <span style="color: #cccccc; margin-left: auto;">{status['next_event_text']}</span>
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Live Prediction", "Model Performance"])

with tab1:
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
            
            # Hide non-trading periods (weekends and nights)
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]), # hide weekends
                    dict(values=["2025-12-25", "2026-01-01"]) # hide holidays (example)
                ]
            )
            
            fig.update_layout(
                title=f"{selected_ticker} Intraday Price (Last 5 Days)", 
                xaxis_rangeslider_visible=False,
                height=500,
                margin=dict(l=0, r=0, t=30, b=0)
            )
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
            df_features = create_features(df)
            
            # Get prediction
            try:
                prediction = predict_next_hour(model, df_features, ticker=selected_ticker)
                current_price = df['Close'].iloc[-1]
                
                # Calculate Target Time
                last_time = df.index[-1]
                target_time = last_time + timedelta(hours=1)
                
                # Format time nicely (e.g., "3:00 PM Today")
                now_day = last_time.date()
                target_day = target_time.date()
                
                if now_day == target_day:
                    day_str = "Today"
                else:
                    day_str = target_time.strftime("%A") # e.g. Monday
                    
                time_str = target_time.strftime("%I:%M %p")
                target_time_display = f"{time_str} {day_str}"
                
                st.metric(label="Current Price", value=f"${current_price:.2f}")
                st.metric(label="Predicted Next Hour Close", value=f"${prediction:.2f}", delta=f"{prediction - current_price:.2f}")
                
                st.info(f"🎯 **Target Time:** {target_time_display}\n\n(Based on data from {last_time.strftime('%I:%M %p')})")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

with tab2:
    st.subheader("Model Accuracy Over Time")
    
    if model is not None and not df.empty:
        with st.spinner("Calculating historical performance..."):
            results, metrics, daily_metrics = evaluate_model(model, df, ticker=selected_ticker)
            
        if not results.empty:
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall MAE", f"${metrics['MAE']:.2f}", help="Mean Absolute Error: Average dollar error per prediction.")
            m2.metric("Directional Accuracy", f"{metrics['Directional_Accuracy']:.1%}", help="Percentage of times the model correctly predicted the price direction (Up/Down).")
            m3.metric("Correct Predictions", f"{metrics['Correct_Count']} / {metrics['Total_Count']}", help="Count of correct direction predictions.")
            
            st.markdown("---")
            
            # Daily Performance Tracker
            st.markdown("### 📅 Daily Performance Tracker")
            st.caption("Breakdown of how accurate the model was for each trading day.")
            
            # Format the daily metrics for display
            daily_display = daily_metrics.copy()
            daily_display.index.name = "Date"
            daily_display['Accuracy'] = daily_display['Accuracy'].apply(lambda x: f"{x:.1%}")
            daily_display['MAE'] = daily_display['MAE'].apply(lambda x: f"${x:.2f}")
            daily_display['Correct / Total'] = daily_display.apply(lambda x: f"{int(x['Correct'])} / {int(x['Total'])}", axis=1)
            daily_display = daily_display[['Accuracy', 'MAE', 'Correct / Total']].sort_index(ascending=False)
            
            st.dataframe(daily_display, use_container_width=True)
            
            st.markdown("---")
            
            # Graph 1: Actual vs Predicted
            st.markdown("### 1. Actual vs Predicted (Last 5 Days)")
            st.caption("This graph compares the **Predicted Close** (orange dashed line) with the **Actual Close** (blue line) that happened 60 minutes later. A perfect model would have the lines overlapping perfectly.")
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=results.index, y=results['Actual'], mode='lines', name='Actual Close (T+60)', line=dict(color='#00B4D8', width=2)))
            fig_perf.add_trace(go.Scatter(x=results.index, y=results['Predicted'], mode='lines', name='Predicted Close', line=dict(color='#FF9F1C', dash='dash', width=2)))
            
            fig_perf.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            fig_perf.update_layout(xaxis_title="Time", yaxis_title="Price", height=500, hovermode="x unified")
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Graph 2: Rolling Accuracy
            st.markdown("### 2. Rolling Accuracy (60-min MAE)")
            st.caption("This graph shows the **Average Error** in dollars over the last 60 minutes. **Lower is better.** Spikes indicate periods where the model struggled (e.g., high volatility news events).")
            
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=results.index, y=results['Rolling_MAE'], mode='lines', name='Rolling MAE', line=dict(color='#EF476F', width=2), fill='tozeroy'))
            
            fig_acc.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            fig_acc.update_layout(xaxis_title="Time", yaxis_title="Error ($)", height=400, hovermode="x unified")
            st.plotly_chart(fig_acc, use_container_width=True)
            
        else:
            st.warning("Not enough data to calculate performance.")
    else:
        st.warning("Model or data not available.")

st.markdown("---")
st.markdown("### Model Info")
if model:
    st.write(f"Model Type: LightGBM Regressor")

# How to Use Section (Expander or separate area)
with st.expander("ℹ️ How to Use & Disclaimer"):
    st.markdown("""
    ### ⚠️ Disclaimer: Not Financial Advice
    **This tool is for informational and educational purposes only.** Do not use this as the sole basis for any investment decisions. The predictions are based on historical patterns and cannot guarantee future results. Financial markets are inherently risky.
    
    ### 🎯 How to Interpret for Prediction Markets
    This tool is designed to help you make informed guesses for **hourly prediction markets** (like Kalshi, Webull, or Polymarket).
    
    **Scenario: "Will SPX close above $6600 at 2 PM?"**
    
    1.  **Check the Prediction:** Look at the "Predicted Next Hour Close".
    2.  **Compare:** 
        *   If **Predicted > Target** (e.g., $6610 > $6600), the model suggests the price will be **UP**. You might consider buying "Yes" contracts.
        *   If **Predicted < Target** (e.g., $6590 < $6600), the model suggests the price will be **DOWN**. You might consider buying "No" contracts.
    3.  **Verify Confidence:** Check the "Model Performance" tab. If the "Directional Accuracy" is high (>55-60%) and the "Rolling MAE" is low, the signal is stronger.
    """)
    # Could add feature importance plot here
