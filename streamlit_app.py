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
from src.model import load_model, predict_next_hour, calculate_probability, get_recent_rmse
from src.azure_logger import log_prediction
from datetime import timedelta



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
    # 1. Data Fetching & Prediction (Do this first so we have variables for the UI)
    with st.spinner("Analyzing Market..."):
        # Fetch data
        # Default to 5d for calculation, but we can use the selector for the chart later if needed
        # Actually, let's keep the selector but maybe move it near the chart or just default to 3d for now
        df = fetch_data(ticker=selected_ticker, period="5d", interval="1m")
        model = load_model(ticker=selected_ticker)
        
    if df.empty:
        st.error("Could not load market data.")
    elif model is None:
        st.warning(f"Model for {selected_ticker} not found. Please train the model.")
    else:
        # Prepare features & Predict
        try:
            # Always calculate prediction so we can show the Edge Finder (Strikes)
            df_features = create_features(df)
            prediction = predict_next_hour(model, df_features, ticker=selected_ticker)
            current_price = df['Close'].iloc[-1]
            rmse = get_recent_rmse(model, df, ticker=selected_ticker)
            
            # Time calculations
            last_time = df.index[-1]
            target_time = last_time + timedelta(hours=1)
            now_day = last_time.date()
            target_day = target_time.date()
            day_str = "Today" if now_day == target_day else target_time.strftime("%A")
            time_str = target_time.strftime("%I:%M %p")
            target_time_display = f"{time_str} {day_str}"
            
            # Check market status for UI customization
            market_status = get_market_status()
            
            if not market_status['is_open']:
                st.warning("😴 **Market is Closed.** Showing analysis based on last closing price.")
            
            # --- TOP SECTION: ACTIONABLE INSIGHTS ---
            # Layout: Left (Metrics + Calculator), Right (Edge Table)
            top_col1, top_col2 = st.columns([1, 1.5])
            
            with top_col1:
                st.subheader("🎯 Prediction")
                st.metric(label="Current Price", value=f"${current_price:.2f}")
                
                # Conditionally show the specific prediction number
                if market_status['is_open']:
                    st.metric(label="Predicted Close", value=f"${prediction:.2f}", delta=f"{prediction - current_price:.2f}")
                    st.caption(f"Target: {target_time_display}")
                else:
                    st.info("Prediction Value Hidden (Market Closed)")
                    
                st.caption(f"Model Uncertainty (RMSE): ±${rmse:.2f}")
                
                with st.expander("ℹ️ How it works"):
                    st.markdown("""
                    **The Model:** Uses LightGBM to predict the exact closing price of the next hour based on recent price action and technical indicators.
                    
                    **RMSE:** The "margin of error". If RMSE is ±$5, the model thinks the price is likely within $5 of the prediction.
                    """)
                
                st.markdown("---")
                st.subheader("🧮 Calculator")
                # Interactive Calculator
                base_price = round(current_price / 10) * 10
                user_strike = st.number_input("Check Strike Price", value=float(base_price), step=5.0)
                if user_strike:
                    user_prob = calculate_probability(prediction, user_strike, rmse)
                    st.metric(f"Prob > ${user_strike}", f"{user_prob:.1f}%")
                    
                    if user_prob > 60:
                        st.success("High Probability of YES")
                    elif user_prob < 40:
                        st.error("High Probability of NO")
                    else:
                        st.warning("Uncertain / Toss-up")
                        
                with st.expander("ℹ️ How to use"):
                    st.markdown("""
                    **Custom Probability:** Enter any strike price (e.g., from Kalshi or Webull) to see the model's calculated probability of the price closing **ABOVE** that level.
                    """)

            with top_col2:
                st.subheader("⚡ Live Opportunities")
                
                # Generate Strikes
                strikes = []
                for i in range(-2, 3): # -20, -10, 0, +10, +20
                    strikes.append(base_price + (i * 10))
                
                edge_data = []
                for strike in strikes:
                    prob_yes = calculate_probability(prediction, strike, rmse)
                    
                    # Simulate Market Price
                    import random
                    noise = random.uniform(-10, 10)
                    market_price_cents = min(99, max(1, int(prob_yes + noise)))
                    
                    edge = prob_yes - market_price_cents
                    
                    if prob_yes > 60 and edge > 5:
                        action = "🟢 BUY YES"
                    elif prob_yes < 40 and edge < -5:
                        action = "🔴 BUY NO"
                    else:
                        action = "⚪ PASS"
                        
                    edge_data.append({
                        "Time": time_str, # Add Time Column
                        "Strike": f"> ${strike}",
                        "Mkt Price": f"{market_price_cents}¢",
                        "Model %": f"{prob_yes:.1f}%",
                        "Edge": f"{edge:.1f}%",
                        "Action": action
                    })
                
                st.table(edge_data)
                
                # Log to Azure
                if market_status['is_open']:
                    log_prediction(prediction, current_price, rmse, edge_data, ticker=selected_ticker)
                
                with st.expander("ℹ️ How to read this table"):
                    st.markdown("""
                    **The Edge Finder:** Compares the Model's Probability against the Market Price (Simulated).
                    
                    *   **Edge:** The difference between our probability and the market's price. Positive edge means the contract is "cheap" relative to our model's confidence.
                    
                    *   **Action:** 
                        *   **BUY YES:** Model is confident price will go HIGHER than strike, and market price is low.
                        *   **BUY NO:** Model is confident price will stay LOWER, and market price for 'Yes' is too high.
                    """)

            # --- BOTTOM SECTION: CONTEXT (CHART) ---
            st.markdown("---")
            st.subheader(f"📉 Market Context ({selected_ticker})")
            
            # Timeframe Selector for Chart
            timeframe = st.radio("Timeframe", ["1 Day", "3 Days", "5 Days"], index=1, horizontal=True, key="chart_timeframe")
            period_map = {"1 Day": "1d", "3 Days": "3d", "5 Days": "5d"}
            selected_period = period_map[timeframe]
            
            # Filter df based on selected_period roughly
            if selected_period == "1d":
                chart_df = df[df.index >= df.index[-1] - timedelta(days=1)]
            elif selected_period == "3d":
                chart_df = df[df.index >= df.index[-1] - timedelta(days=3)]
            else:
                chart_df = df # 5d was the default fetch
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=chart_df.index,
                            open=chart_df['Open'],
                            high=chart_df['High'],
                            low=chart_df['Low'],
                            close=chart_df['Close'],
                            name=selected_ticker))
            
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(values=["2025-12-25", "2026-01-01"]),
                    dict(pattern="hour", bounds=[16, 9.5])
                ]
            )
            
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=400,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in analysis: {e}")

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
