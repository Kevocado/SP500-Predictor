import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
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

# Azure Logger with Safety Wrapper
try:
    from src.azure_logger import log_prediction, fetch_all_logs
    AZURE_AVAILABLE = True
except Exception as e:
    AZURE_AVAILABLE = False
    print(f"⚠️ Azure logging disabled: {e}")

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

# Title Bar
col_title, col_refresh = st.columns([3, 1])
with col_title:
    st.title("⚡ Prediction Market Edge Finder")
with col_refresh:
    if st.button("🔄 Refresh Data", use_container_width=True):
        run_scanner()
        st.rerun()

st.markdown("---")

# --- PILLS NAVIGATION (Asset Selector) ---
st.markdown("### Select Asset")
selected_ticker = st.pills(
    "Asset Selection",
    options=["SPX", "Nasdaq", "BTC", "ETH"],
    default="SPX" if 'selected_asset' not in st.session_state else st.session_state.get('selected_asset', 'SPX'),
    label_visibility="collapsed",
    key="asset_pills"
)

# Update session state
if st.session_state.get('last_selected_asset') != selected_ticker:
    st.session_state.last_selected_asset = selected_ticker
    st.session_state.selected_asset = selected_ticker
    
    # Smart Timeframe Logic
    recommended_tf = determine_best_timeframe(selected_ticker)
    reverse_map = {"Hourly": "Hourly", "Daily": "End of Day"}
    st.session_state.timeframe_view = reverse_map[recommended_tf]
    
    if recommended_tf == "Daily" and get_market_status(selected_ticker)['is_open'] == False:
        st.toast(f"Switched to End of Day view (Market Closed for {selected_ticker})", icon="ℹ️")

# Determine timeframe (using smart logic, no UI selector needed for now)
recommended_tf = determine_best_timeframe(selected_ticker)
timeframe_map = {"Hourly": "Hourly", "Daily": "Daily"}
internal_timeframe = timeframe_map.get(recommended_tf, "Daily")

# Sidebar: Keep only essential controls
if st.sidebar.button("🔄 Retrain Model"):
    with st.status(f"Retraining model for {selected_ticker}...", expanded=True) as status:
        st.write("Fetching data...")
        try:
            df = fetch_data(ticker=selected_ticker, period="7d", interval="1m")
            st.write("Processing features...")
            df = prepare_training_data(df)
            st.write("Training model...")
            train_model(df, ticker=selected_ticker)
            status.update(label="Model retrained successfully!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")

# --- MAIN CONTENT AREA ---
# We'll have: Alpha Deck → Scanner Table → (Inline Deep Dive when row selected)
# And a separate Performance tab

# === ALPHA DECK (Hero Section) ===
st.markdown("### 💎 Alpha Deck")

all_strikes = st.session_state.scan_results['strikes']
all_ranges = st.session_state.scan_results['ranges']

# Filter for selected asset only
asset_strikes = [s for s in all_strikes if s['Asset'] == selected_ticker]

if asset_strikes:
    # Calculate metrics for the selected asset
    
    # Metric 1: Best Edge (highest confidence = furthest from 50%)
    # Edge = how far the probability is from 50% (uncertainty)
    best_edge_strike = max(asset_strikes, key=lambda x: abs(float(x['Prob'].strip('%')) - 50))
    best_edge_val = abs(float(best_edge_strike['Prob'].strip('%')) - 50)
    
    # Metric 2: Highest Confidence (probability closest to 0 or 100)
    highest_conf_strike = max(asset_strikes, key=lambda x: abs(x['Numeric_Prob'] - 50))
    highest_conf_val = highest_conf_strike['Numeric_Prob']
    # Cap at 99.9%
    highest_conf_val = min(highest_conf_val, 99.9)
    
    # Metric 3: Market Mover (for now, show current prediction vs current price %)
    # We'll need to fetch this from the current data
    try:
        if internal_timeframe == "Daily":
            df_alpha = fetch_data(ticker=selected_ticker, period="60d", interval="1h")
            model_alpha = load_daily_model(ticker=selected_ticker)
            if not df_alpha.empty and model_alpha:
                df_features_alpha, _ = prepare_daily_data(df_alpha)
                pred_alpha = predict_daily_close(model_alpha, df_features_alpha.iloc[[-1]])
                curr_alpha = df_alpha['Close'].iloc[-1]
                move_pct = ((pred_alpha - curr_alpha) / curr_alpha) * 100
            else:
                move_pct = 0
        else:
            df_alpha = fetch_data(ticker=selected_ticker, period="5d", interval="1m")
            model_alpha = load_model(ticker=selected_ticker)
            if not df_alpha.empty and model_alpha:
                df_features_alpha = create_features(df_alpha)
                pred_alpha = predict_next_hour(model_alpha, df_features_alpha, ticker=selected_ticker)
                curr_alpha = df_alpha['Close'].iloc[-1]
                move_pct = ((pred_alpha - curr_alpha) / curr_alpha) * 100
            else:
                move_pct = 0
    except:
        move_pct = 0
    
    # Display 3 metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💎 Best Edge",
            value=f"{best_edge_val:.1f}% Confidence",
            delta=best_edge_strike['Strike'],
            help=f"Highest confidence opportunity: {best_edge_strike['Action']}"
        )
    
    with col2:
        st.metric(
            label="🛡️ Highest Confidence",
            value=f"{highest_conf_val:.1f}%",
            delta=highest_conf_strike['Strike'],
            help=f"Most confident prediction: {highest_conf_strike['Action']}"
        )
    
    with col3:
        move_label = "⚡ Predicted Move"
        move_direction = "↑" if move_pct > 0 else "↓" if move_pct < 0 else "→"
        st.metric(
            label=move_label,
            value=f"{move_direction} {abs(move_pct):.2f}%",
            delta=f"{internal_timeframe}",
            help="Expected price movement for next timeframe"
        )
else:
    st.info(f"No opportunities found for {selected_ticker}. Try refreshing data.")

st.markdown("---")

# === SCANNER TABLE (Master) ===
st.markdown("### 📊 Live Scanner")

# Filter strikes for selected asset
asset_strikes_table = [s for s in all_strikes if s['Asset'] == selected_ticker]

if asset_strikes_table:
    df_strikes = pd.DataFrame(asset_strikes_table)
    
    # SORTING: Best Probability (highest confidence)
    df_strikes['Edge_Abs'] = abs(df_strikes['Numeric_Prob'] - 50)
    df_strikes = df_strikes.sort_values('Edge_Abs', ascending=False)
    
    # Add index column for selection
    df_strikes = df_strikes.reset_index(drop=True)
    df_strikes.insert(0, 'Select', df_strikes.index)
    
    # Display columns
    cols = ['Select', 'Timeframe', 'Date', 'Time', 'Strike', 'Prob', 'Action']
    
    def highlight_edge(row):
        prob = float(row['Prob'].strip('%'))
        if prob > 70:
            return ['background-color: #1b4d1b'] * len(row)
        elif prob < 30:
            return ['background-color: #4d1b1b'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df_strikes[cols].style.apply(highlight_edge, axis=1), 
        use_container_width=True,
        hide_index=True
    )
    
    # Row selection input
    st.markdown("---")
    selected_row_idx = st.number_input(
        "🔍 Select Row for Deep Dive (enter number from 'Select' column)",
        min_value=0,
        max_value=len(df_strikes) - 1,
        value=0,
        step=1,
        key="row_selector"
    )
    
    # === INLINE DEEP DIVE (Detail) ===
    if selected_row_idx is not None and selected_row_idx < len(df_strikes):
        selected_strike = df_strikes.iloc[selected_row_idx]
        
        with st.expander(f"🔍 Deep Dive: {selected_strike['Action']} at {selected_strike['Strike']}", expanded=True):
            st.markdown(f"### Analysis for {selected_strike['Asset']} - {selected_strike['Timeframe']}")
            st.caption(f"Target: {selected_strike['Date']} at {selected_strike['Time']}")
            
            # Fetch fresh data for bell curve
            try:
                if selected_strike['Timeframe'] == "Daily":
                    df_deep = fetch_data(ticker=selected_ticker, period="60d", interval="1h")
                    model_deep = load_daily_model(ticker=selected_ticker)
                    if not df_deep.empty and model_deep:
                        df_features_deep, _ = prepare_daily_data(df_deep)
                        pred_deep = predict_daily_close(model_deep, df_features_deep.iloc[[-1]])
                        rmse_deep = df_deep['Close'].iloc[-1] * 0.01
                        curr_price_deep = df_deep['Close'].iloc[-1]
                    else:
                        raise Exception("No data")
                else:
                    df_deep = fetch_data(ticker=selected_ticker, period="5d", interval="1m")
                    model_deep = load_model(ticker=selected_ticker)
                    if not df_deep.empty and model_deep:
                        df_features_deep = create_features(df_deep)
                        pred_deep = predict_next_hour(model_deep, df_features_deep, ticker=selected_ticker)
                        rmse_deep = get_recent_rmse(model_deep, df_deep, ticker=selected_ticker)
                        curr_price_deep = df_deep['Close'].iloc[-1]
                    else:
                        raise Exception("No data")
                
                # Parse strike price from the selected row
                strike_str = selected_strike['Strike']
                # Strike format is typically "$5,500" or similar
                strike_price = float(strike_str.replace('$', '').replace(',', ''))
                
                # Calculate probability using the model's prediction distribution
                prob_val = calculate_probability(pred_deep, rmse_deep, strike_price, selected_strike['Action'])
                
                # Create bell curve visualization
                st.markdown("#### 📊 Probability Distribution")
                
                # Generate x-axis (price range)
                x_min = pred_deep - 4 * rmse_deep
                x_max = pred_deep + 4 * rmse_deep
                x = np.linspace(x_min, x_max, 500)
                
                # Normal distribution PDF
                y = stats.norm.pdf(x, pred_deep, rmse_deep)
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Add distribution curve
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    fill='tozeroy',
                    name='Probability Distribution',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                # Highlight the region based on action
                if selected_strike['Action'] == 'Call':
                    # Shade area above strike
                    mask = x >= strike_price
                    fig.add_trace(go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='lines',
                        fill='tozeroy',
                        name=f'Prob Above {strike_str}',
                        line=dict(color='#1b4d1b', width=0),
                        fillcolor='rgba(27, 77, 27, 0.3)'
                    ))
                else:  # Put
                    # Shade area below strike
                    mask = x <= strike_price
                    fig.add_trace(go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='lines',
                        fill='tozeroy',
                        name=f'Prob Below {strike_str}',
                        line=dict(color='#4d1b1b', width=0),
                        fillcolor='rgba(77, 27, 27, 0.3)'
                    ))
                
                # Add vertical lines for prediction, current price, and strike
                fig.add_vline(x=pred_deep, line_dash="dash", line_color="yellow", annotation_text="Predicted", annotation_position="top")
                fig.add_vline(x=curr_price_deep, line_dash="dash", line_color="white", annotation_text="Current", annotation_position="top")
                fig.add_vline(x=strike_price, line_color="red", line_width=3, annotation_text="Strike", annotation_position="top")
                
                fig.update_layout(
                    title=f"Probability: {selected_strike['Prob']}",
                    xaxis_title="Price",
                    yaxis_title="Probability Density",
                    height=400,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show key metrics
                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.metric("Current Price", f"${curr_price_deep:,.2f}")
                col_d2.metric("Predicted Price", f"${pred_deep:,.2f}", f"{((pred_deep - curr_price_deep) / curr_price_deep * 100):+.2f}%")
                col_d3.metric("Model Uncertainty (σ)", f"${rmse_deep:,.2f}")
                
            except Exception as e:
                st.error(f"Unable to load Deep Dive data: {e}")
        
else:
    st.info(f"No strike opportunities found for {selected_ticker}. Try refreshing data.")

st.markdown("---")

# === HELP SECTION ===
with st.expander("📘 Help & Strategy"):
    st.markdown("""
    ### How to use this tool
    1.  **Check the Alpha Deck:** The top 3 metrics show the best opportunities for the selected asset.
    2.  **Scan the Table:** Review all strikes sorted by confidence.
    3.  **Deep Dive:** Click a row to see detailed probability analysis.
    
    ### Strategy Guide
    *   **Best Edge:** The strike with the highest edge (model prob vs market)  
    *   **Highest Confidence:** The most certain prediction
    *   **Predicted Move:** Expected price change for the next timeframe
    """)

# === TABS: Main Page is default, Performance is separate ===
main_tab, perf_tab = st.tabs(["📊 Trading Dashboard", "📈 Model Performance & History"])

with main_tab:
    # This tab is empty because all content is already on the main page above
    # The trading content (Alpha Deck + Scanner) is rendered before the tabs
    st.info("👆 All trading information is displayed above. Use the Performance tab to review model accuracy and history.")

with perf_tab:
    st.markdown("### 📈 Model Performance & Analytics")
    st.caption(f"Historical performance metrics for {selected_ticker}")
    
    # Fetch data for performance analysis
    try:
        if internal_timeframe == "Daily":
            df_perf = fetch_data(ticker=selected_ticker, period="60d", interval="1h")
            model_perf = load_daily_model(ticker=selected_ticker)
        else:
            df_perf = fetch_data(ticker=selected_ticker, period="5d", interval="1m")
            model_perf = load_model(ticker=selected_ticker)
    except:
        df_perf = pd.DataFrame()
        model_perf = None
    
    if model_perf is not None and not df_perf.empty:
        with st.spinner("Calculating historical performance..."):
            try:
                results, metrics, daily_metrics = evaluate_model(model_perf, df_perf, ticker=selected_ticker)
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
    else:
        st.warning("Model not found or no data available for performance analysis.")
    
    st.markdown("---")
    
    # === AZURE LOGS ===
    st.markdown("### 📜 Historical Logs (Azure)")
    st.caption("Immutable audit trail of all predictions made by this system.")
    
    # Check Azure availability
    if not AZURE_AVAILABLE:
        st.warning("⚠️ Azure Logging disabled: Credentials not found. Add AZURE_CONNECTION_STRING to .env or Streamlit secrets.")
    else:
        try:
            df_logs = fetch_all_logs()
            if not df_logs.empty:
                st.dataframe(df_logs.sort_values('timestamp_utc', ascending=False), use_container_width=True)
            else:
                st.info("No logs found in Azure container yet.")
        except Exception as e:
            st.error(f"❌ Error fetching logs: {str(e)[:100]}")

# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and does not constitute financial advice. Trading involves risk.")
