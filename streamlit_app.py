import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import timedelta, time

# Load environment variables with explicit path
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
load_dotenv(dotenv_path=env_path)

# Hybrid Loading: .env (Local/Backend) vs st.secrets (Streamlit Cloud)
# If AZURE_CONNECTION_STRING is missing (e.g., on Cloud where .env is gitignored),
# try to load it from Streamlit Secrets into os.environ for compatibility.
if not os.getenv("AZURE_CONNECTION_STRING"):
    try:
        if "AZURE_CONNECTION_STRING" in st.secrets:
            os.environ["AZURE_CONNECTION_STRING"] = st.secrets["AZURE_CONNECTION_STRING"]
            print("✅ Loaded AZURE_CONNECTION_STRING from st.secrets (Cloud Mode)")
    except FileNotFoundError:
        # st.secrets not found (local without .streamlit/secrets.toml)
        pass

# Debug Status
if os.getenv("AZURE_CONNECTION_STRING"):
    print("✅ Azure Connection String is SET.")
else:
    print("❌ Azure Connection String is MISSING. Check .env or secrets.toml.")

# Add src to path so we can import modules
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import fetch_data
from src.feature_engineering import create_features, prepare_training_data
from src.model import load_model, predict_next_hour, train_model, calculate_probability, get_recent_rmse
from src.evaluation import evaluate_model
from src.utils import get_market_status, determine_best_timeframe
from src.model_daily import load_daily_model, predict_daily_close, prepare_daily_data
from src.signals import generate_trading_signals
from src.azure_logger import log_prediction, fetch_all_logs

st.set_page_config(page_title="Prediction Market Edge Finder", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {'strikes': [], 'ranges': []}
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = "SPX"

# --- HELPER FUNCTIONS ---
def run_scanner(timeframe_override=None):
    """
    Runs the market scanner and updates session state.
    """
    tickers_to_scan = ["SPX", "Nasdaq", "BTC", "ETH"]
    all_strikes = []
    all_ranges = []
    
    # Use the override if provided, else use the current view logic (which might be tricky in a loop)
    # Actually, the scanner should probably scan based on the "Best" timeframe for each asset?
    # Or should it respect a global "Scan Mode"?
    # The prompt says: "When the user switches assets... automatically switch the Timeframe toggle".
    # But for the *Scanner* (which shows ALL assets), what timeframe does it uses?
    # Ideally, it scans each asset in its BEST timeframe.
    # Let's implement "Smart Scanning": Scan each asset in its optimal timeframe.
    
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers_to_scan):
        try:
            # Determine Timeframe for this asset
            # If user forced a timeframe in the UI, maybe we should use that?
            # But the scanner shows everything. Let's use the "Smart" timeframe for each asset.
            # OR, if the user selected "Daily" globally, maybe they want Daily for everything?
            # Let's stick to the "Smart" logic for the scanner to ensure relevance.
            
            # Actually, to keep it simple and consistent with the UI toggle:
            # We will use the `determine_best_timeframe` for each asset individually.
            best_tf = determine_best_timeframe(ticker)
            
            # Check Status
            market_status = get_market_status(ticker)
            if not market_status['is_open'] and best_tf == "Hourly":
                # If it's closed and we wanted Hourly, we can't really do much unless we switch to Daily.
                # determine_best_timeframe handles this: if Closed -> Daily.
                pass
            
            # Fetch Data
            if best_tf == "Daily":
                df_scan = fetch_data(ticker=ticker, period="60d", interval="1h")
                model_scan = load_daily_model(ticker=ticker)
            else:
                df_scan = fetch_data(ticker=ticker, period="5d", interval="1m")
                model_scan = load_model(ticker=ticker)
                
            if df_scan.empty or not model_scan: continue
            
            # Predict
            if best_tf == "Daily":
                df_features_scan, _ = prepare_daily_data(df_scan)
                pred_scan = predict_daily_close(model_scan, df_features_scan.iloc[[-1]])
                rmse_scan = df_scan['Close'].iloc[-1] * 0.01 # Approx RMSE
                
                last_time = df_scan.index[-1]
                target_time = last_time.replace(hour=16, minute=0, second=0, microsecond=0)
                if last_time.time() >= time(16, 0):
                    target_time += timedelta(days=1)
            else:
                df_features_scan = create_features(df_scan)
                pred_scan = predict_next_hour(model_scan, df_features_scan, ticker=ticker)
                rmse_scan = get_recent_rmse(model_scan, df_scan, ticker=ticker)
                
                last_time = df_scan.index[-1]
                target_time = last_time + timedelta(hours=1)
            
            curr_price_scan = df_scan['Close'].iloc[-1]
            
            date_str = target_time.strftime("%b %d")
            time_str = target_time.strftime("%I:%M %p")
            
            # Generate Signals
            signals = generate_trading_signals(ticker, pred_scan, curr_price_scan, rmse_scan)
            
            for s in signals['strikes']:
                s['Asset'] = ticker
                s['Date'] = date_str
                s['Time'] = time_str
                s['Timeframe'] = best_tf # Add this so user knows
                s['Numeric_Prob'] = float(s['Prob'].strip('%'))
                all_strikes.append(s)
                
            for r in signals['ranges']:
                r['Asset'] = ticker
                r['Date'] = date_str
                r['Time'] = time_str
                r['Timeframe'] = best_tf
                all_ranges.append(r)
                
        except Exception as e:
            print(f"Scanner error on {ticker}: {e}")
        
        progress_bar.progress((i + 1) / len(tickers_to_scan))
        
    progress_bar.empty()
    st.session_state.scan_results = {'strikes': all_strikes, 'ranges': all_ranges}
    st.session_state.last_scan_time = pd.Timestamp.now()

# --- AUTO-RUN SCANNER ON LOAD ---
if not st.session_state.scan_results['strikes'] and not st.session_state.scan_results['ranges']:
    run_scanner()

# --- LAYOUT ---

# Top Bar: Title & Refresh
col_title, col_refresh = st.columns([3, 1])
with col_title:
    st.title("⚡ Prediction Market Edge Finder")
with col_refresh:
    if st.button("🔄 Refresh Data", use_container_width=True):
        run_scanner()
        st.rerun()

# Top Navigation (Asset Selector)
st.markdown("### Select Asset")
selected_ticker = st.radio(
    "Select Asset", 
    ["SPX", "Nasdaq", "BTC", "ETH"], 
    horizontal=True,
    label_visibility="collapsed",
    key="asset_selector"
)

# Smart Timeframe Logic
# We want to update the timeframe based on the selected asset, BUT only if the asset CHANGED.
# Since Streamlit reruns on interaction, we can check if selected_ticker changed.
if st.session_state.get('last_selected_asset') != selected_ticker:
    # Asset changed!
    recommended_tf = determine_best_timeframe(selected_ticker)
    # Map internal "Daily" back to UI "End of Day"
    reverse_map = {"Hourly": "Hourly", "Daily": "End of Day"}
    st.session_state.timeframe_view = reverse_map[recommended_tf]
    st.session_state.last_selected_asset = selected_ticker
    
    if recommended_tf == "Daily" and get_market_status(selected_ticker)['is_open'] == False:
        st.toast(f"Switched to End of Day view (Market Closed for {selected_ticker})", icon="ℹ️")

# Timeframe Selector (Sidebar or Top?)
# User asked to remove Sidebar selector. Let's put Timeframe near the Nav or in Sidebar?
# "Top-Level Navigation (Replace Sidebar Selector)" -> Refers to Asset.
# Let's keep Timeframe in Sidebar for now or move it to top right?
# Sidebar is fine for settings, but "Smart Timeframe" implies it's active.
# Let's put it in the sidebar but controllable.
# Timeframe Selector
timeframe_view = st.sidebar.radio(
    "Timeframe", 
    ["Hourly", "End of Day"], 
    key="timeframe_view", # Linked to session state
    help="Hourly: Predicts price 60 mins from now.\nEnd of Day: Predicts closing price at 4:00 PM ET."
)

# Map "End of Day" back to "Daily" for internal logic if needed, or just use "End of Day" string
# Let's normalize it to "Daily" for internal logic to avoid breaking everything
timeframe_map = {"Hourly": "Hourly", "End of Day": "Daily"}
internal_timeframe = timeframe_map[timeframe_view]

if st.sidebar.button("Retrain Model"):
    with st.status(f"Retraining model for {selected_ticker}...", expanded=True) as status:
        st.write("Fetching data...")
        # Logic for retraining (omitted for brevity, keeping existing logic if needed or just placeholder)
        # Re-implementing the retraining logic briefly
        try:
            df = fetch_data(ticker=selected_ticker, period="7d", interval="1m")
            st.write("Processing features...")
            df = prepare_training_data(df)
            st.write("Training model...")
            train_model(df, ticker=selected_ticker)
            status.update(label="Model retrained successfully!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Live Scanner", "🔍 Deep Dive", "📈 Model Performance", "📜 History"])

with tab1:
    # --- ALPHA DECK ---
    all_strikes = st.session_state.scan_results['strikes']
    all_ranges = st.session_state.scan_results['ranges']
    
    if all_strikes:
        # Sort by Probability (Confidence)
        # We want the highest CONFIDENCE. 
        # Confidence = abs(Prob - 50). 99% is high confidence, 1% is high confidence (that it won't happen).
        
        # Group by Asset to get the best one for EACH instrument
        best_per_asset = {}
        for opp in all_strikes:
            asset = opp['Asset']
            confidence = abs(opp['Numeric_Prob'] - 50)
            
            if asset not in best_per_asset:
                best_per_asset[asset] = (opp, confidence)
            else:
                if confidence > best_per_asset[asset][1]:
                    best_per_asset[asset] = (opp, confidence)
        
        # Extract the opportunities
        top_opps = [val[0] for val in best_per_asset.values()]
        # Sort them by confidence globally just for display order
        top_opps.sort(key=lambda x: abs(x['Numeric_Prob'] - 50), reverse=True)
        
        st.markdown("### 🔥 Top Opportunities")
        
        # Dynamic columns based on how many assets found (max 4)
        cols = st.columns(len(top_opps))
        
        for i, col in enumerate(cols):
            opp = top_opps[i]
            
            # Display Logic: If BUY NO, show 100 - Prob as "Win Prob"
            if "NO" in opp['Action']:
                win_prob = 100 - opp['Numeric_Prob']
                display_prob = f"{win_prob:.1f}%"
            else:
                display_prob = opp['Prob']
                
            with col:
                st.markdown(f"""
                <div style="padding: 15px; border: 1px solid #333; border-radius: 10px; background-color: #0e1117;">
                    <h3 style="margin:0; color: #3b82f6;">{opp['Asset']} <span style="font-size:0.6em; color:grey">({opp['Timeframe']})</span></h3>
                    <p style="font-size: 1.2em; font-weight: bold; margin: 5px 0;">{opp['Strike']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="background-color: {'#1b4d1b' if 'YES' in opp['Action'] else '#4d1b1b'}; padding: 2px 8px; border-radius: 4px; font-size: 0.9em;">{opp['Action']}</span>
                        <span style="font-weight: bold;">{display_prob}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")

    # --- SCANNER TABLES ---
    scan_tab1, scan_tab2 = st.tabs(["🎯 Strike Prices (Direction)", "📊 Daily Ranges (Volatility)"])
    
    with scan_tab1:
        if all_strikes:
            df_strikes = pd.DataFrame(all_strikes)
            
            # SORTING: Best Probability (closest to 0 or 100, i.e., highest confidence)
            # We added 'Numeric_Prob' earlier. Let's use it.
            # We want rows with Prob > 50 sorted desc, and Prob < 50 sorted asc?
            # Or just sort by "Edge" (distance from 50%)
            df_strikes['Edge_Abs'] = abs(df_strikes['Numeric_Prob'] - 50)
            df_strikes = df_strikes.sort_values('Edge_Abs', ascending=False).drop(columns=['Edge_Abs'])
            
            cols = ['Asset', 'Timeframe', 'Date', 'Time', 'Strike', 'Prob', 'Action']
            
            def highlight_edge(row):
                prob = float(row['Prob'].strip('%'))
                if prob > 70:
                    return ['background-color: #1b4d1b'] * len(row)
                elif prob < 30:
                    return ['background-color: #4d1b1b'] * len(row)
                return [''] * len(row)

            st.dataframe(df_strikes[cols].style.apply(highlight_edge, axis=1), use_container_width=True)
        else:
            st.info("No active strike opportunities found.")
            
    with scan_tab2:
        if all_ranges:
            def highlight_winner(row):
                return ['background-color: #1b4d1b' if row['Is_Winner'] else '' for _ in row]
            
            df_ranges = pd.DataFrame(all_ranges)
            
            # SORTING: Show Winners first
            df_ranges = df_ranges.sort_values('Is_Winner', ascending=False)
            
            cols = ['Asset', 'Timeframe', 'Date', 'Time', 'Range', 'Predicted In Range?', 'Action', 'Is_Winner']
            st.dataframe(df_ranges[cols].style.apply(highlight_winner, axis=1), use_container_width=True)
        else:
            st.info("No range opportunities found.")
            
    # --- HELP EXPANDER ---
    with st.expander("📘 Help & Strategy"):
        st.markdown("""
        ### How to use this tool
        1.  **Check the Alpha Deck:** The top 3 highest-confidence trades are shown at the top.
        2.  **Scan the Market:** Use the **Strikes** tab for directional bets (Up/Down) and the **Ranges** tab for volatility bets (Price Brackets).
        3.  **Deep Dive:** Click the "Deep Dive" tab to analyze a specific asset in detail.
        
        ### Strategy Guide
        *   **Strike Prices:** If the model says **🟢 BUY YES**, it means the probability is significantly higher than 50%.
        *   **Daily Ranges:** Look for the **Green Highlighted** row. This is where the model predicts the price will land.
        *   **Timeframes:**
            *   **Hourly:** Best for short-term speculation (Crypto, Active Market Hours).
            *   **Daily:** Best for "End of Day" closes (Stocks after hours).
        """)

with tab2:
    # --- DEEP DIVE ---
    st.subheader(f"🔍 Deep Dive: {selected_ticker}")
    
    # Market Status
    status = get_market_status(selected_ticker)
    st.markdown(f"**Status:** <span style='color:{status['color']}; font-weight:bold'>{status['status_text']}</span>", unsafe_allow_html=True)
    if not status['is_open']:
        st.caption(f"Next Event: {status['next_event_text']}")
        
    # Fetch Data for Deep Dive
    with st.spinner(f"Analyzing {selected_ticker}..."):
        if internal_timeframe == "Daily":
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
            if internal_timeframe == "Daily":
                df_features, _ = prepare_daily_data(df)
                last_row = df_features.iloc[[-1]]
                prediction = predict_daily_close(model, last_row)
                current_price = df['Close'].iloc[-1]
                rmse = current_price * 0.01 
                
                last_time = df.index[-1]
                target_time = last_time.replace(hour=16, minute=0, second=0, microsecond=0)
                if last_time.time() >= time(16, 0):
                    target_time += timedelta(days=1)
                time_str = "4:00 PM (Close)"
            else:
                df_features = create_features(df)
                prediction = predict_next_hour(model, df_features, ticker=selected_ticker)
                current_price = df['Close'].iloc[-1]
                rmse = get_recent_rmse(model, df, ticker=selected_ticker)
                
                last_time = df.index[-1]
                target_time = last_time + timedelta(hours=1)
                time_str = target_time.strftime("%I:%M %p")
            
            now_day = last_time.date()
            target_day = target_time.date()
            day_str = "Today" if now_day == target_day else target_time.strftime("%A")
            target_time_display = f"{time_str} {day_str}"
            
            # Display Prediction
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:,.2f}")
            col2.metric(f"Predicted {internal_timeframe}", f"${prediction:,.2f}", delta=f"{prediction-current_price:,.2f}")
            col3.metric("Target Time", target_time_display)
            
            # Edge Finder (Deep Dive specific)
            st.subheader("🎯 Edge Finder")
            signals = generate_trading_signals(selected_ticker, prediction, current_price, rmse)
            
            dd_tab1, dd_tab2 = st.tabs(["Strikes", "Ranges"])
            with dd_tab1:
                st.dataframe(pd.DataFrame(signals['strikes']), use_container_width=True)
            with dd_tab2:
                 # Highlight winners
                def highlight_winner(row):
                    return ['background-color: #1b4d1b' if row['Is_Winner'] else '' for _ in row]
                st.dataframe(pd.DataFrame(signals['ranges']).style.apply(highlight_winner, axis=1), use_container_width=True)
                
            # Log to Azure
            log_prediction(prediction, current_price, rmse, signals['strikes'], ticker=selected_ticker)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

with tab3:
    # --- MODEL PERFORMANCE ---
    st.subheader("Model Accuracy Over Time")
    
    if 'model' in locals() and model is not None and not df.empty:
        with st.spinner("Calculating historical performance..."):
            try:
                results, metrics, daily_metrics = evaluate_model(model, df, ticker=selected_ticker)
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
                results = pd.DataFrame()
            
        if not results.empty:
            # --- TRUST ENGINE METRICS ---
            st.markdown("### 🛡️ The Trust Engine")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Directional Accuracy", f"{metrics['Directional_Accuracy']:.1%}", help="How often the model correctly predicts Up vs Down.")
            m2.metric("Brier Score", f"{metrics['Brier_Score']:.3f}", help="Probabilistic Error. 0.0 is perfect, 0.25 is random guessing. Lower is better.")
            m3.metric("Total PnL (Sim)", f"${metrics['Total_PnL']:,.0f}", help="Simulated Profit/Loss if betting $100 on high-confidence signals (>60% or <40%).")
            m4.metric("MAE", f"${metrics['MAE']:.2f}", help="Average dollar error per prediction.")
            
            st.markdown("---")
            
            # --- CHARTS ---
            col_charts1, col_charts2 = st.columns(2)
            
            with col_charts1:
                st.markdown("#### 📈 Cumulative PnL (Backtest)")
                st.caption("Growth of a $10,000 account betting $100 per trade.")
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(x=results.index, y=results['Cum_PnL'], mode='lines', name='PnL', fill='tozeroy', line=dict(color='#3b82f6')))
                fig_pnl.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_pnl, use_container_width=True)
                
            with col_charts2:
                st.markdown("#### 🎯 Calibration Curve")
                st.caption("Does 70% probability actually mean 70% win rate? (Ideal: Diagonal Line)")
                
                prob_true = metrics['Calibration_Data']['prob_true']
                prob_pred = metrics['Calibration_Data']['prob_pred']
                
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ideal', line=dict(color='grey', dash='dash')))
                fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model', line=dict(color='#1b4d1b')))
                fig_cal.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Predicted Probability", yaxis_title="Actual Win Rate")
                st.plotly_chart(fig_cal, use_container_width=True)
            
            st.markdown("---")
            
            # Daily Performance Tracker
            st.markdown("### 📅 Daily Performance Tracker")
            
            daily_display = daily_metrics.copy()
            daily_display.index.name = "Date"
            daily_display['Accuracy'] = daily_display['Accuracy'].apply(lambda x: f"{x:.1%}")
            daily_display['Daily PnL'] = daily_display['Daily_PnL'].apply(lambda x: f"${x:,.0f}")
            daily_display['MAE'] = daily_display['MAE'].apply(lambda x: f"${x:.2f}")
            daily_display = daily_display[['Accuracy', 'Daily PnL', 'MAE']].sort_index(ascending=False)
            
            st.dataframe(daily_display, use_container_width=True)

with tab4:
    st.subheader("📜 Historical Logs (Azure)")
    st.caption("Immutable audit trail of all predictions made by this system.")
    
    # Check connection string
    if not os.getenv("AZURE_CONNECTION_STRING"):
        st.error("❌ AZURE_CONNECTION_STRING not found in .env file.")
    else:
        df_logs = fetch_all_logs()
        if not df_logs.empty:
            st.dataframe(df_logs.sort_values('timestamp_utc', ascending=False), use_container_width=True)
        else:
            st.info("No logs found in Azure container yet.")

# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and does not constitute financial advice. Trading involves risk.")
