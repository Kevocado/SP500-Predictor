import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import altair as alt
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

# Azure Logger with Safety Wrapper (Only needed for main app if we log predictions here, but for now we keep it if needed or remove if unused. 
# The prompt said "Move all code related to... Azure Logs". 
# But we might still need logging for predictions? 
# The prompt said "Cut all the code related to 'Model Performance', 'Trust Engine', 'Calibration Curve', and 'Azure Logs'".
# It didn't explicitly say remove logging of NEW predictions. 
# However, let's keep the import if it's used for logging new predictions, but remove the "Historical Logs" section at the bottom.
# Actually, looking at the code, `log_prediction` is imported but not used in the visible code? 
# Ah, `log_prediction` is likely used inside `generate_trading_signals` or similar? No, it's usually called after prediction.
# Let's check if `log_prediction` is used. It's imported at line 48.
# Searching the file... it's NOT used in the provided code!
# So we can remove the import block entirely if we want to be clean, or just leave it.
# Let's remove the unused imports to be clean, as per "Move all code related to... Azure Logs".

# Removing unused imports for analytics


st.set_page_config(page_title="Prediction Market Edge Finder", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {'strikes': [], 'ranges': []}
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = "SPX"

# --- HELPER FUNCTIONS ---
def create_probability_bar(model_prob, market_prob=50):
    """
    Creates a simple horizontal bar chart showing Model Prob vs Market Prob (Breakeven).
    """
    # Data for the chart
    data = pd.DataFrame({
        'Probability': [model_prob],
        'Label': ['Model']
    })
    
    # Base chart
    base = alt.Chart(data).encode(
        x=alt.X('Probability', scale=alt.Scale(domain=[0, 100]), axis=None),
        y=alt.Y('Label', axis=None)
    )
    
    # The Bar (Model Probability)
    # Color logic: Green if > Market, Red if < Market (but usually we show "Confidence" > 50, so mostly Green?)
    # Actually, let's stick to the "Edge" logic. 
    # If we are betting YES, and Model > Market, it's Green.
    # If we are betting NO, and Model < Market, it's Green (for the NO bet).
    # Let's just make it Blue for "Model Prediction" and use the Card Text color for Sentiment.
    # Or use the user's request: "Green if Model > Market, Red if Model < Market".
    
    color = "#1b4d1b" if model_prob > market_prob else "#4d1b1b"
    
    bar = base.mark_bar(color=color, height=20)
    
    # The Rule (Market/Breakeven Probability)
    rule = alt.Chart(pd.DataFrame({'x': [market_prob]})).mark_rule(color='white', strokeWidth=2).encode(x='x')
    
    return (bar + rule).properties(height=30, width='container')

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
        # --- 1. DAILY SCAN (End of Day) ---
        # Run for ALL assets
        try:
            df_daily = fetch_data(ticker=ticker, period="60d", interval="1h")
            model_daily = load_daily_model(ticker=ticker)
            
            if not df_daily.empty and model_daily:
                df_features_daily, _ = prepare_daily_data(df_daily)
                pred_daily = predict_daily_close(model_daily, df_features_daily.iloc[[-1]])
                rmse_daily = df_daily['Close'].iloc[-1] * 0.01 # Approx RMSE
                curr_price_daily = df_daily['Close'].iloc[-1]
                
                last_time = df_daily.index[-1]
                target_time = last_time.replace(hour=16, minute=0, second=0, microsecond=0)
                if last_time.time() >= time(16, 0):
                    target_time += timedelta(days=1)
                    
                date_str = target_time.strftime("%b %d")
                time_str = target_time.strftime("%I:%M %p")
                
                signals_daily = generate_trading_signals(ticker, pred_daily, curr_price_daily, rmse_daily)
                
                for s in signals_daily['strikes']:
                    s['Asset'] = ticker
                    s['Date'] = date_str
                    s['Time'] = time_str
                    s['Timeframe'] = "Daily"
                    s['Numeric_Prob'] = float(s['Prob'].strip('%'))
                    s['RMSE'] = rmse_daily
                    all_strikes.append(s)
                    
                # Ranges from Daily? Maybe not needed if Hourly covers it, but let's add if useful.
                # Usually ranges are better for short term volatility.
                
        except Exception as e:
            print(f"Daily Scanner error on {ticker}: {e}")

        # --- 2. HOURLY SCAN (Intraday) ---
        # Run if market is OPEN or it's Crypto (24/7)
        market_status = get_market_status(ticker)
        is_crypto = ticker in ["BTC", "ETH"]
        
        if market_status['is_open'] or is_crypto:
            try:
                df_hourly = fetch_data(ticker=ticker, period="5d", interval="1m")
                model_hourly = load_model(ticker=ticker)
                
                if not df_hourly.empty and model_hourly:
                    df_features_hourly = create_features(df_hourly)
                    pred_hourly = predict_next_hour(model_hourly, df_features_hourly, ticker=ticker)
                    rmse_hourly = get_recent_rmse(model_hourly, df_hourly, ticker=ticker)
                    curr_price_hourly = df_hourly['Close'].iloc[-1]
                    
                    last_time = df_hourly.index[-1]
                    target_time = last_time + timedelta(hours=1)
                    
                    date_str = target_time.strftime("%b %d")
                    time_str = target_time.strftime("%I:%M %p")
                    
                    signals_hourly = generate_trading_signals(ticker, pred_hourly, curr_price_hourly, rmse_hourly)
                    
                    for s in signals_hourly['strikes']:
                        s['Asset'] = ticker
                        s['Date'] = date_str
                        s['Time'] = time_str
                        s['Timeframe'] = "Hourly"
                        s['Numeric_Prob'] = float(s['Prob'].strip('%'))
                        s['RMSE'] = rmse_hourly
                        all_strikes.append(s)
                        
                    for r in signals_hourly['ranges']:
                        r['Asset'] = ticker
                        r['Date'] = date_str
                        r['Time'] = time_str
                        r['Timeframe'] = "Hourly"
                        all_ranges.append(r)
            except Exception as e:
                print(f"Hourly Scanner error on {ticker}: {e}")
        
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
    # FIX: Normalize to Win Probability (0-100% Confidence)
    # If Prob > 50% -> Confidence = Prob (BUY YES)
    # If Prob < 50% -> Confidence = 100 - Prob (BUY NO)
    
    def get_confidence_and_signal(strike_data):
        prob = strike_data['Numeric_Prob']
        if prob > 50:
            return prob, "BUY YES"
        else:
            return 100 - prob, "BUY NO"

    highest_conf_strike = max(asset_strikes, key=lambda x: get_confidence_and_signal(x)[0])
    highest_conf_val, conf_signal = get_confidence_and_signal(highest_conf_strike)
    
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
    
    # Display 3 metric cards using containers for better control
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.caption("💎 Best Edge")
            # Action Color
            action_color = "green" if "BUY YES" in best_edge_strike['Action'] else "red"
            st.markdown(f"### :{action_color}[{best_edge_strike['Action']}]")
            st.markdown(f"**{best_edge_strike['Strike']}**")
            
            # Probability Bar
            # Assuming Market Prob is ~50% for ATM, but ideally we'd know the option price.
            # For now, use 50% as the "Breakeven" anchor or just visualize the Model Prob.
            # User asked for: "Marker at Market Price". We don't have live option prices, so 50% is a fair proxy for "Unknown".
            st.altair_chart(create_probability_bar(best_edge_strike['Numeric_Prob'], 50), use_container_width=True)
            
            c_footer1, c_footer2 = st.columns(2)
            c_footer1.caption(f"Conf: **{best_edge_val:.1f}%**")
            c_footer2.caption(f"Edge: **{abs(best_edge_strike['Numeric_Prob'] - 50):.1f}%**")
            
            if st.button("🔍 View Bell Curve", key="btn_alpha_edge"):
                st.session_state.selected_strike = best_edge_strike
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.caption("🛡️ Highest Confidence")
            action_color = "green" if "BUY YES" in conf_signal else "red"
            st.markdown(f"### :{action_color}[{conf_signal}]")
            st.markdown(f"**{highest_conf_strike['Strike']}**")
            
            st.altair_chart(create_probability_bar(highest_conf_strike['Numeric_Prob'], 50), use_container_width=True)
            
            c_footer1, c_footer2 = st.columns(2)
            c_footer1.caption(f"Conf: **{highest_conf_val:.1f}%**")
            c_footer2.caption(f"Prob: **{highest_conf_strike['Numeric_Prob']:.1f}%**")
            
            if st.button("🔍 View Bell Curve", key="btn_alpha_conf"):
                st.session_state.selected_strike = highest_conf_strike
                st.rerun()
    
    with col3:
        with st.container(border=True):
            st.caption("⚡ Predicted Move")
            move_direction = "↑" if move_pct > 0 else "↓" if move_pct < 0 else "→"
            move_color = "green" if move_pct > 0 else "red" if move_pct < 0 else "gray"
            st.markdown(f"### :{move_color}[{move_direction} {abs(move_pct):.2f}%]")
            st.markdown(f"**{internal_timeframe}**")
            
            # For move, maybe a bar showing relative strength? Or just text.
            # Let's keep it simple as text for now, or a centered bar?
            # User asked for "Probability Bar" for "top 3 opportunities". 
            # This one is a "Move", not a "Trade". 
            # Let's just show the price targets cleanly.
            st.write("") # Spacer to align with charts
            st.write("") 
            
            c_footer1, c_footer2 = st.columns(2)
            c_footer1.caption(f"Target: **${pred_alpha:,.2f}**")
            c_footer2.caption(f"Current: **${curr_alpha:,.2f}**")
            
            # No "Deep Dive" for the Move card, or maybe link to the asset generally?
            # Let's leave it as info only.

else:
    st.info(f"No opportunities found for {selected_ticker}. Try refreshing data.")

st.markdown("---")

# === DEEP DIVE SECTION (Master Detail) ===
if 'selected_strike' in st.session_state and st.session_state.selected_strike:
    selected_strike = st.session_state.selected_strike
    # Ensure it matches current asset to avoid confusion
    if selected_strike['Asset'] == selected_ticker:
        st.markdown("---")
        with st.expander(f"🔍 Deep Dive: {selected_strike['Action']} at {selected_strike['Strike']}", expanded=True):
            st.markdown(f"### Analysis for {selected_strike['Asset']} - {selected_strike['Timeframe']}")
            st.caption(f"Target: {selected_strike['Date']} at {selected_strike['Time']}")
            
            # Fetch fresh data for bell curve
            try:
                if selected_strike['Timeframe'] == "Daily" or selected_strike['Timeframe'] == "End of Day":
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
                strike_str = str(selected_strike['Strike'])
                # Clean the string: Remove '>', '$', and whitespace
                strike_price = float(strike_str.replace('>', '').replace('$', '').replace(',', '').strip())
                
                # Calculate probability using the model's prediction distribution
                # FIX: Removed 4th argument (current_price) which caused TypeError
                prob_val = calculate_probability(pred_deep, strike_price, rmse_deep)
                
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
                if "BUY YES" in selected_strike['Action']:
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
                else:  # BUY NO
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

st.markdown("---")

# === OPPORTUNITY BOARD (Sectioned Cards) ===
st.markdown("### 📋 Trade Opportunity Board")

# Filter strikes for selected asset
asset_strikes_board = [s for s in all_strikes if s['Asset'] == selected_ticker]
asset_ranges_board = [r for r in all_ranges if r['Asset'] == selected_ticker]

# Tabs for Sections
tab_hourly, tab_daily, tab_ranges = st.tabs(["⚡ Hourly Snipes", "📅 Daily Close", "🎯 Ranges"])

with tab_hourly:
    st.caption("Short-term opportunities expiring in < 60 mins")
    hourly_ops = [s for s in asset_strikes_board if s['Timeframe'] == "Hourly"]
    
    if hourly_ops:
        # Sort by Edge (Confidence)
        hourly_ops.sort(key=lambda x: abs(x['Numeric_Prob'] - 50), reverse=True)
        
        for i, op in enumerate(hourly_ops):
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                
                # Reliability Badge Logic
                rmse_val = op.get('RMSE', 15.0)
                # SPX/NDX have high RMSE (10-30), others lower. 
                # User rule: < 5.0 High Precision, > 10.0 Volatile.
                # We'll stick to that strictly.
                if rmse_val < 5.0:
                    badge = "✅ High Precision Model"
                elif rmse_val > 10.0:
                    badge = "⚠️ Volatile Model"
                else:
                    badge = "ℹ️ Normal Volatility"
                
                with c1:
                    st.markdown(f"### {op['Strike']}")
                    st.caption(f"Expires: {op['Time']}")
                
                with c2:
                    # Calculate Edge/Confidence for display
                    prob = op['Numeric_Prob']
                    if prob > 50:
                        conf = prob
                        signal = "BUY YES"
                    else:
                        conf = 100 - prob
                        signal = "BUY NO"
                    
                    st.metric("Confidence", f"{conf:.1f}%", f"{signal}")
                    st.caption(badge)
                
                with c3:
                    st.write("") # Spacer
                    if st.button("Deep Dive", key=f"dd_h_{i}_{op['Strike']}"):
                        st.session_state.selected_strike = op
                        st.rerun()
    else:
        st.info("No Hourly opportunities found.")

with tab_daily:
    st.caption("End-of-day predictions (4:00 PM Close)")
    daily_ops = [s for s in asset_strikes_board if s['Timeframe'] == "Daily" or s['Timeframe'] == "End of Day"]
    
    if daily_ops:
        daily_ops.sort(key=lambda x: abs(x['Numeric_Prob'] - 50), reverse=True)
        
        for i, op in enumerate(daily_ops):
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                
                # Reliability Badge
                rmse_val = op.get('RMSE', 15.0)
                if rmse_val < 5.0:
                    badge = "✅ High Precision Model"
                elif rmse_val > 10.0:
                    badge = "⚠️ Volatile Model"
                else:
                    badge = "ℹ️ Normal Volatility"
                
                with c1:
                    st.markdown(f"### {op['Strike']}")
                    st.caption(f"Target: {op['Date']} Close")
                
                with c2:
                    prob = op['Numeric_Prob']
                    if prob > 50:
                        conf = prob
                        signal = "BUY YES"
                    else:
                        conf = 100 - prob
                        signal = "BUY NO"
                    
                    st.metric("Confidence", f"{conf:.1f}%", f"{signal}")
                    st.caption(badge)
                
                with c3:
                    st.write("")
                    if st.button("Deep Dive", key=f"dd_d_{i}_{op['Strike']}"):
                        st.session_state.selected_strike = op
                        st.rerun()
    else:
        st.info("No Daily opportunities found.")

with tab_ranges:
    st.caption("Volatility plays: Will price stay within range?")
    if asset_ranges_board:
        for i, r in enumerate(asset_ranges_board):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {r['Range']}")
                    st.write(f"Predicted In Range: **{r['Predicted In Range?']}**")
                with c2:
                    st.caption(f"Action: {r['Action']}")
    else:
        st.info("No Range opportunities found.")


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

# === TABS REMOVED ===
# Analytics moved to pages/1_📈_Performance.py


# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and does not constitute financial advice. Trading involves risk.")
