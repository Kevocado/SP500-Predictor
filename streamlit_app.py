import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os

# Load environment variables with explicit path and override IMMEDIATELY
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import altair as alt
import sys
from datetime import timedelta, time

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
from src.kalshi_feed import get_real_kalshi_markets, check_kalshi_connection
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
    
    # Use accessible colors for Light/Dark mode
    color = "#22c55e" if model_prob > market_prob else "#ef4444"
    
    bar = base.mark_bar(color=color, height=20)
    
    # The Rule (Market/Breakeven Probability)
    # Use a color that stands out in both modes (e.g., Gray or Black/White mix? Or just Orange?)
    # White is invisible in Light Mode. Black is invisible in Dark Mode.
    # Let's use a safe Gray or Orange.
    rule = alt.Chart(pd.DataFrame({'x': [market_prob]})).mark_rule(color='gray', strokeWidth=3).encode(x='x')
    
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
                # Generate Signals (Simulated for now, but we will overlay Kalshi data if available)
                signals_daily = generate_trading_signals(ticker, pred_daily, curr_price_daily, rmse_daily)
            
                # --- KALSHI OVERLAY ---
                # Fetch real markets
                real_markets = get_real_kalshi_markets(ticker)
                # Create a lookup for real prices: {strike_price: {'yes': ..., 'no': ...}}
                # Note: Strike matching might be tricky due to float precision or exact values.
                # We'll try to find the closest match or exact match.
                real_market_map = {}
                for rm in real_markets:
                    if rm.get('strike_price'):
                        real_market_map[float(rm['strike_price'])] = rm
                
                for s in signals_daily['strikes']:
                    s['Asset'] = ticker
                    s['Date'] = date_str
                    s['Time'] = time_str
                    s['Timeframe'] = "Daily" 
                    s['Numeric_Prob'] = float(s['Prob'].strip('%'))
                    s['RMSE'] = rmse_daily
                    
                    # Overlay Real Data if found
                    # s['Strike'] is a string like "> $5,920". Need to parse number.
                    try:
                        strike_val = float(s['Strike'].replace('>','').replace('<','').replace('$','').replace(',','').strip())
                        # Find match in real_market_map (exact or very close)
                        # For now, simple exact match or closest?
                        # Let's just check if it exists directly first.
                        if strike_val in real_market_map:
                            rm = real_market_map[strike_val]
                            s['Real_Yes_Bid'] = rm['yes_bid']
                            s['Real_No_Bid'] = rm['no_bid']
                            # Recalculate Edge based on Real Price?
                            # If Action is BUY YES: Edge = Model_Prob - Yes_Bid
                            # If Action is BUY NO: Edge = (100 - Model_Prob) - No_Bid (Wait, No_Bid is cost to buy No)
                            # Actually: Edge = Model_Prob_of_Winning - Cost_of_Bet
                            
                            model_prob_win = s['Numeric_Prob'] if "BUY YES" in s['Action'] else (100 - s['Numeric_Prob'])
                            cost = rm['yes_bid'] if "BUY YES" in s['Action'] else rm['no_bid']
                            
                            # If cost is in cents (1-99), convert to % (0.01-0.99) or keep as is?
                            # Model prob is 0-100. Cost is usually 1-99 cents.
                            # So Edge = Model_Prob - Cost.
                            if cost > 0:
                                s['Real_Edge'] = model_prob_win - cost
                                s['Has_Real_Data'] = True
                            else:
                                s['Has_Real_Data'] = False
                        else:
                            s['Has_Real_Data'] = False
                    except:
                        s['Has_Real_Data'] = False
                    
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

if st.session_state.last_scan_time:
    st.caption(f"Last Updated: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

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
st.sidebar.markdown("### 🔌 System Status")

# Check Kalshi Connection (Real Check)
if check_kalshi_connection():
    api_status = "✅ Kalshi Live"
else:
    api_status = "❌ Error"
    
st.sidebar.caption(f"API Status: **{api_status}**")

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
            
            # Probability Bar (Normalized)
            # Logic: Bar Value = Probability of WINNING the recommended bet.
            # If BUY YES -> Model Prob.
            # If BUY NO -> 100 - Model Prob.
            # Color is always GREEN because it's a "Good Bet" (High Confidence).
            
            prob_win = best_edge_strike['Numeric_Prob'] if "BUY YES" in best_edge_strike['Action'] else (100 - best_edge_strike['Numeric_Prob'])
            
            # Distance Metric & Context
            try:
                strike_val = float(best_edge_strike['Strike'].replace('>','').replace('$','').replace(',','').strip())
                dist_pct = ((strike_val - curr_alpha) / curr_alpha) * 100
                
                # OTM/ITM Logic
                # For Binary Call (> Strike):
                # Price < Strike -> OTM (Needs to go UP)
                # Price > Strike -> ITM (Already winning)
                is_itm = curr_alpha > strike_val
                status_icon = "🎯" if is_itm else "📉"
                status_label = "ITM" if is_itm else "OTM"
                dist_abs = abs(strike_val - curr_alpha)
                
                context_str = f"Current: ${curr_alpha:,.0f} {status_icon} ${dist_abs:.0f} {status_label}"
            except:
                context_str = "Context N/A"

            st.altair_chart(create_probability_bar(prob_win, 50), use_container_width=True)
            st.caption(context_str)
            
            c_footer1, c_footer2 = st.columns(2)
            c_footer1.caption(f"Conf: **{prob_win:.1f}%**")
            # Edge is Model Prob - 50 (roughly)
            edge_val = abs(best_edge_strike['Numeric_Prob'] - 50)
            c_footer2.caption(f"Edge: **{edge_val:.1f}%**")
            
            if st.button("🔍 View Bell Curve", key="btn_alpha_edge"):
                st.session_state.selected_strike = best_edge_strike
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.caption("🛡️ Highest Confidence")
            action_color = "green" if "BUY YES" in conf_signal else "red"
            st.markdown(f"### :{action_color}[{conf_signal}]")
            st.markdown(f"**{highest_conf_strike['Strike']}**")
            
            # Probability Bar (Normalized)
            prob_win_conf = highest_conf_strike['Numeric_Prob'] if "BUY YES" in conf_signal else (100 - highest_conf_strike['Numeric_Prob'])
            
            # Context
            try:
                strike_val_c = float(highest_conf_strike['Strike'].replace('>','').replace('$','').replace(',','').strip())
                is_itm_c = curr_alpha > strike_val_c
                status_icon_c = "🎯" if is_itm_c else "📉"
                status_label_c = "ITM" if is_itm_c else "OTM"
                dist_abs_c = abs(strike_val_c - curr_alpha)
                
                context_str_c = f"Current: ${curr_alpha:,.0f} {status_icon_c} ${dist_abs_c:.0f} {status_label_c}"
            except:
                context_str_c = "Context N/A"

            st.altair_chart(create_probability_bar(prob_win_conf, 50), use_container_width=True)
            st.caption(context_str_c)
            
            c_footer1, c_footer2 = st.columns(2)
            c_footer1.caption(f"Conf: **{prob_win_conf:.1f}%**")
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
                        line=dict(color='#22c55e', width=0),
                        fillcolor='rgba(34, 197, 94, 0.3)' # Green with opacity
                    ))
                else:  # BUY NO
                    # Shade area below strike
                    mask = x <= strike_price
                    fig.add_trace(go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='lines',
                        fill='tozeroy',
                        name=f'Prob Below {strike_str}',
                        line=dict(color='#ef4444', width=0),
                        fillcolor='rgba(239, 68, 68, 0.3)' # Red with opacity
                    ))
                
                # Add vertical lines for prediction, current price, and strike
                fig.add_vline(x=pred_deep, line_dash="dash", line_color="orange", annotation_text="Predicted", annotation_position="top")
                # Current price needs to be visible in both modes. 'gray' is usually safe.
                fig.add_vline(x=curr_price_deep, line_dash="dash", line_color="gray", annotation_text="Current", annotation_position="top")
                fig.add_vline(x=strike_price, line_color="#ef4444", line_width=3, annotation_text="Strike", annotation_position="top")
                
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
                # Deal Ticket Layout: 2 Columns
                c1, c2 = st.columns([1.5, 1])
                
                # Logic Prep
                prob = op['Numeric_Prob']
                if prob > 50:
                    conf = prob
                    signal = "BUY YES"
                    is_buy_yes = True
                else:
                    conf = 100 - prob
                    signal = "BUY NO"
                    is_buy_yes = False
                
                # Context Logic
                try:
                    strike_val = float(op['Strike'].replace('>','').replace('$','').replace(',','').strip())
                    # Use curr_alpha if available and matching
                    curr = curr_alpha if op['Asset'] == selected_ticker else 0
                    
                    if curr > 0:
                        # OTM/ITM Logic
                        # Buy NO: Want Price < Strike. 
                        # If Price < Strike -> OTM (Safe/Winning).
                        # If Price > Strike -> ITM (Risk/Losing).
                        # Buy YES: Want Price > Strike.
                        # If Price > Strike -> ITM (Winning).
                        # If Price < Strike -> OTM (Losing).
                        
                        if not is_buy_yes: # BUY NO
                            if curr < strike_val:
                                badge_text = "📉 OTM (Safe)"
                                badge_color = "green"
                            else:
                                badge_text = "⚠️ ITM (Risk)"
                                badge_color = "red"
                        else: # BUY YES
                            if curr > strike_val:
                                badge_text = "🎯 ITM (Winning)"
                                badge_color = "green"
                            else:
                                badge_text = "📉 OTM (Losing)"
                                badge_color = "red"
                                
                        context_line = f"Current: ${curr:,.0f}"
                    else:
                        badge_text = "Waiting for Data"
                        badge_color = "gray"
                        context_line = "Current: N/A"
                except:
                    badge_text = "N/A"
                    badge_color = "gray"
                    context_line = "Current: N/A"

                with c1:
                    # Header: Signal
                    st.markdown(f"### :green[{signal}]")
                    # Sub-header: Strike
                    st.markdown(f"**{op['Strike']}**")
                    # Badge
                    st.caption(f":{badge_color}[{badge_text}]")
                    # Context
                    st.caption(context_line)
                
                with c2:
                    # Financials
                    # Bid/Ask
                    if op.get('Has_Real_Data'):
                        bid = op.get('Real_Yes_Bid') if is_buy_yes else op.get('Real_No_Bid')
                        # Ask is usually Bid + Spread. Kalshi spread is tight, maybe +1-2 cents?
                        # We don't have Ask in the feed currently, let's just show Bid.
                        # Or if we want to be fancy, Bid / Ask placeholder.
                        # User asked for "Bid: 42¢ | Ask: 45¢". We only have Bid from our feed.
                        # Let's just show Bid for now or update feed later.
                        price_str = f"Bid: {bid}¢"
                    else:
                        price_str = "Bid: --"
                    
                    st.markdown(f"**{price_str}**")
                    
                    # Model Value
                    model_val = int(conf) # roughly cents
                    st.caption(f"Model: {model_val}¢")
                    
                    # Edge
                    edge = abs(conf - 50) # Rough edge proxy or use Real Edge
                    if op.get('Has_Real_Data'):
                        real_edge = op.get('Real_Edge', 0)
                        edge_str = f"🔥 +{real_edge:.1f}% Edge"
                    else:
                        edge_str = f"⚡ {edge:.1f}% Conf"
                        
                    st.markdown(f"**{edge_str}**")
                    
                    if st.button("Analyze", key=f"dd_h_{i}_{op['Strike']}"):
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
                # Deal Ticket Layout: 2 Columns
                c1, c2 = st.columns([1.5, 1])
                
                # Logic Prep
                prob = op['Numeric_Prob']
                if prob > 50:
                    conf = prob
                    signal = "BUY YES"
                    is_buy_yes = True
                else:
                    conf = 100 - prob
                    signal = "BUY NO"
                    is_buy_yes = False
                
                # Context Logic
                try:
                    strike_val = float(op['Strike'].replace('>','').replace('$','').replace(',','').strip())
                    curr = curr_alpha if op['Asset'] == selected_ticker else 0
                    
                    if curr > 0:
                        if not is_buy_yes: # BUY NO
                            if curr < strike_val:
                                badge_text = "📉 OTM (Safe)"
                                badge_color = "green"
                            else:
                                badge_text = "⚠️ ITM (Risk)"
                                badge_color = "red"
                        else: # BUY YES
                            if curr > strike_val:
                                badge_text = "🎯 ITM (Winning)"
                                badge_color = "green"
                            else:
                                badge_text = "📉 OTM (Losing)"
                                badge_color = "red"
                                
                        context_line = f"Current: ${curr:,.0f}"
                    else:
                        badge_text = "Waiting for Data"
                        badge_color = "gray"
                        context_line = "Current: N/A"
                except:
                    badge_text = "N/A"
                    badge_color = "gray"
                    context_line = "Current: N/A"

                with c1:
                    st.markdown(f"### :green[{signal}]")
                    st.markdown(f"**{op['Strike']}**")
                    st.caption(f":{badge_color}[{badge_text}]")
                    st.caption(context_line)
                
                with c2:
                    if op.get('Has_Real_Data'):
                        bid = op.get('Real_Yes_Bid') if is_buy_yes else op.get('Real_No_Bid')
                        price_str = f"Bid: {bid}¢"
                    else:
                        price_str = "Bid: --"
                    
                    st.markdown(f"**{price_str}**")
                    
                    model_val = int(conf)
                    st.caption(f"Model: {model_val}¢")
                    
                    edge = abs(conf - 50)
                    if op.get('Has_Real_Data'):
                        real_edge = op.get('Real_Edge', 0)
                        edge_str = f"🔥 +{real_edge:.1f}% Edge"
                    else:
                        edge_str = f"⚡ {edge:.1f}% Conf"
                        
                    st.markdown(f"**{edge_str}**")
                    
                    if st.button("Analyze", key=f"dd_d_{i}_{op['Strike']}"):
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
