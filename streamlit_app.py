import streamlit as st
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import fetch_data
from src.feature_engineering import create_features, prepare_training_data
from src.model import load_model, predict_next_hour, train_model

st.set_page_config(page_title="Prediction Market Edge Finder", layout="wide")

st.title("🔮 Prediction Market Edge Finder")
st.markdown("""
**Hourly Volatility & Probability Engine**
This tool identifies "mispriced risk" in hourly prediction markets (e.g., Kalshi, ForecastEx). 
It calculates the probability of price targets for Indices, Crypto, and High-Vol Stocks.
""")

# Sidebar for controls
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Select Asset", ["SPX", "Nasdaq", "BTC", "ETH"])
timeframe_view = st.sidebar.radio("Timeframe", ["Hourly", "Daily"], index=0)

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
from src.model_daily import load_daily_model, predict_daily_close, prepare_daily_data
from src.signals import generate_trading_signals
from src.azure_logger import log_prediction, fetch_all_logs
from datetime import timedelta

def check_daily_range(predicted_price, ranges_list):
    """
    Checks which range the predicted price falls into.
    
    Args:
        predicted_price (float): The predicted closing price.
        ranges_list (list): List of tuples (min, max).
        
    Returns:
        tuple: The matching range (min, max) or None.
    """
    for r in ranges_list:
        if r[0] <= predicted_price < r[1]:
            return r
    return None

# Main content
st.title(f"📈 {selected_ticker} Hourly Predictor")

tab1, tab2, tab3 = st.tabs(["Live Prediction", "Model Performance", "📜 History (Azure)"])

with tab1:
    # 1. Market Scanner (Top of Page)
    st.subheader("🚨 Live Market Scanner")
    
    if st.button("Scan All Markets"):
        scanner_progress = st.progress(0)
        
        # Store results for both tabs
        all_strikes = []
        all_ranges = []
        
        tickers_to_scan = ["SPX", "Nasdaq", "BTC", "ETH"]
        
        for i, ticker in enumerate(tickers_to_scan):
            try:
                # Check Status FIRST
                market_status = get_market_status(ticker)
                if not market_status['is_open']:
                    continue # Skip closed markets

                # Fetch & Predict
                # Use Daily model for Scanner if Timeframe is Daily? 
                # The prompt implies "Daily Range" is a key feature.
                # Let's stick to the selected timeframe or maybe scan both?
                # For simplicity, let's respect the global timeframe_view selector
                
                if timeframe_view == "Daily":
                    df_scan = fetch_data(ticker=ticker, period="60d", interval="1h")
                    model_scan = load_daily_model(ticker=ticker)
                else:
                    df_scan = fetch_data(ticker=ticker, period="5d", interval="1m")
                    model_scan = load_model(ticker=ticker)
                    
                if df_scan.empty or not model_scan: continue
                
                # Predict
                if timeframe_view == "Daily":
                    df_features_scan, _ = prepare_daily_data(df_scan)
                    pred_scan = predict_daily_close(model_scan, df_features_scan.iloc[[-1]])
                    rmse_scan = df_scan['Close'].iloc[-1] * 0.01 # Approx RMSE
                else:
                    df_features_scan = create_features(df_scan)
                    pred_scan = predict_next_hour(model_scan, df_features_scan, ticker=ticker)
                    rmse_scan = get_recent_rmse(model_scan, df_scan, ticker=ticker)
                
                curr_price_scan = df_scan['Close'].iloc[-1]
                
                # Generate Signals
                signals = generate_trading_signals(ticker, pred_scan, curr_price_scan, rmse_scan)
                
                # Add Ticker info to signals
                for s in signals['strikes']:
                    s['Asset'] = ticker
                    all_strikes.append(s)
                    
                for r in signals['ranges']:
                    r['Asset'] = ticker
                    all_ranges.append(r)
                    
            except Exception as e:
                print(f"Scanner error on {ticker}: {e}")
            
            scanner_progress.progress((i + 1) / len(tickers_to_scan))
            
        scanner_progress.empty()
        
        # Display Results in Tabs
        scan_tab1, scan_tab2 = st.tabs(["🎯 Strike Prices (Direction)", "📊 Daily Ranges (Volatility)"])
        
        with scan_tab1:
            if all_strikes:
                st.caption(f"Directional opportunities based on {timeframe_view} prediction.")
                st.dataframe(all_strikes, use_container_width=True)
            else:
                st.info("No active strike opportunities found.")
                
        with scan_tab2:
            if all_ranges:
                st.caption(f"Range bucket opportunities based on {timeframe_view} prediction.")
                
                # Highlight winners
                def highlight_winner(row):
                    return ['background-color: #1b4d1b' if row['Is_Winner'] else '' for _ in row]
                
                df_ranges = pd.DataFrame(all_ranges)
                # Drop the boolean column for display if desired, or keep it
                # Applying style
                st.dataframe(df_ranges.style.apply(highlight_winner, axis=1), use_container_width=True)
            else:
                st.info("No range opportunities found.")
            
    with st.expander("ℹ️ Guide: Which Timeframe to use?"):
        st.markdown("""
        *   **Hourly (Standard):** Use this for standard prediction markets (e.g., "Will BTC be > X at 2:00 PM?"). The model predicts the price 60 minutes from now.
        *   **Daily (Close):** Use this for "Daily Close" markets (e.g., "Will SPX close > X today?"). The model predicts the price at 4:00 PM ET.
        """)
            
    st.markdown("---")

    # 2. Deep Dive (Single Asset)
    st.subheader(f"🔍 Deep Dive: {selected_ticker}")
    
    # Market Status Indicator
    status = get_market_status(selected_ticker)
    st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {'#1b4d1b' if status['is_open'] else '#4d1b1b'}; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
            <span style="height: 12px; width: 12px; background-color: {status['color']}; border-radius: 50%; display: inline-block;"></span>
            <span style="font-weight: bold; color: white;">{status['status_text']}</span>
            <span style="color: #cccccc; margin-left: auto;">{status['next_event_text']}</span>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Analyzing {selected_ticker}..."):
        # Fetch data
        # For Daily model, we need hourly data (1h)
        # For Hourly model, we need minute data (1m)
        if timeframe_view == "Daily":
            df = fetch_data(ticker=selected_ticker, period="60d", interval="1h")
            model = load_daily_model(ticker=selected_ticker)
        else:
            df = fetch_data(ticker=selected_ticker, period="5d", interval="1m")
            model = load_model(ticker=selected_ticker)
        
    if df.empty:
        st.error("Could not load market data.")
    elif model is None:
        st.warning(f"Model for {selected_ticker} not found. Please train the model.")
    else:
        # Prepare features & Predict
        try:
            if timeframe_view == "Daily":
                # Daily Model Logic
                df_features, _ = prepare_daily_data(df)
                # Use the last row for prediction
                last_row = df_features.iloc[[-1]]
                prediction = predict_daily_close(model, last_row)
                current_price = df['Close'].iloc[-1]
                # RMSE for daily model (approximate from test set or just hardcode a safe buffer for now)
                # Ideally we'd calculate this dynamically like get_recent_rmse
                rmse = current_price * 0.01 # 1% volatility assumption for now
                
                # Time calculations
                last_time = df.index[-1]
                # Target is 4 PM of the same day (or next trading day if past close)
                # Logic handled in prepare_daily_data implicitly by target, but for display:
                target_time = last_time.replace(hour=16, minute=0, second=0, microsecond=0)
                if last_time.time() >= time(16, 0):
                    target_time += timedelta(days=1)
                time_str = "4:00 PM (Close)"
                
            else:
                # Hourly Model Logic
                df_features = create_features(df)
                prediction = predict_next_hour(model, df_features, ticker=selected_ticker)
                current_price = df['Close'].iloc[-1]
                rmse = get_recent_rmse(model, df, ticker=selected_ticker)
                
                # Time calculations
                last_time = df.index[-1]
                target_time = last_time + timedelta(hours=1)
                time_str = target_time.strftime("%I:%M %p")
            
            now_day = last_time.date()
            target_day = target_time.date()
            day_str = "Today" if now_day == target_day else target_time.strftime("%A")
            target_time_display = f"{time_str} {day_str}"
                
            now_day = last_time.date()
            target_day = target_time.date()
            day_str = "Today" if now_day == target_day else target_time.strftime("%A")
            target_time_display = f"{time_str} {day_str}"
            
            # Check market status for UI customization
            market_status = get_market_status(selected_ticker)
            
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
                    st.metric(label=f"Predicted {timeframe_view}", value=f"${prediction:.2f}", delta=f"{prediction - current_price:.2f}")
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
                
                if timeframe_view == "Daily":
                    st.subheader("📊 Daily Range Bucket")
                    # Generate Ranges around current price
                    base = round(current_price / 100) * 100
                    ranges = []
                    for i in range(-2, 3):
                        ranges.append((base + i*100, base + (i+1)*100))
                    
                    matching_range = check_daily_range(prediction, ranges)
                    
                    for r in ranges:
                        is_match = r == matching_range
                        icon = "✅" if is_match else "⬜"
                        st.write(f"{icon} ${r[0]} - ${r[1]}")
                        
                else:
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
                        "Date": now_day.strftime("%b %d"), # Readable Date (e.g. Nov 22)
                        "Time": time_str, 
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

with tab3:
    st.subheader("☁️ Azure Audit Trail & Analytics")
    st.markdown("This dashboard pulls live historical data from your **Azure Data Lake** to monitor model performance in production.")
    
    if st.button("Refresh Data from Azure"):
        st.cache_data.clear()
        
    with st.spinner("Fetching logs from Azure..."):
        history_df = fetch_all_logs()
        
    if not history_df.empty:
        # Filter by selected ticker
        ticker_history = history_df[history_df['ticker'] == selected_ticker].copy()
        
        if not ticker_history.empty:
            # 1. KPI Metrics
            kpi1, kpi2, kpi3 = st.columns(3)
            total_preds = len(ticker_history)
            avg_rmse = ticker_history['model_rmse'].mean()
            
            # Calculate PnL (Simulation)
            # Logic: If Action != PASS, we bet $100.
            # Win condition: If Action is BUY YES, did price > strike?
            # We need ACTUAL price at expiry to calculate real PnL.
            # Since we only log at prediction time, we might not have the result yet.
            # For this MVP, let's just show the "Edge" capture potential or simple stats.
            # Or we can try to match it with historical data if available.
            # Let's stick to "Edge Captured" for now.
            
            avg_edge = ticker_history['best_edge_val'].mean()
            
            kpi1.metric("Total Predictions", total_preds)
            kpi2.metric("Avg Model RMSE", f"${avg_rmse:.2f}")
            kpi3.metric("Avg Edge Found", f"{avg_edge:.1f}%")
            
            st.markdown("---")
            
            # 2. Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("### 📉 Predicted vs Actual (at time of request)")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=ticker_history['timestamp_utc'], y=ticker_history['current_price'], name='Actual Price', line=dict(color='#00B4D8')))
                fig_hist.add_trace(go.Scatter(x=ticker_history['timestamp_utc'], y=ticker_history['predicted_price'], name='Predicted', line=dict(color='#FF9F1C', dash='dash')))
                fig_hist.update_layout(title="Price History", height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col_chart2:
                st.markdown("### 📊 Error Distribution (RMSE)")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=ticker_history['model_rmse'], nbinsx=20, marker_color='#EF476F'))
                fig_dist.update_layout(title="Model Uncertainty Distribution", xaxis_title="RMSE ($)", height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # 3. Raw Data
            with st.expander("View Raw Audit Logs"):
                st.dataframe(ticker_history.sort_values('timestamp_utc', ascending=False), use_container_width=True)
                
        else:
            st.info(f"No history found for {selected_ticker} yet. Make some predictions!")
    else:
        st.warning("No logs found in Azure. Check your connection string or make some predictions first.")
        if not os.getenv("AZURE_CONNECTION_STRING"):
            st.error("⚠️ AZURE_CONNECTION_STRING not found in environment variables. Please check your .env file.")
        else:
            st.caption("Connection string is detected. The container might be empty.")

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
