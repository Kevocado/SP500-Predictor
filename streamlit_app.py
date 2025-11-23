import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os

# Load environment variables with explicit path and override IMMEDIATELY
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import altair as alt
import sys
from datetime import timedelta, time, datetime

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
if 'wager_amount' not in st.session_state:
    st.session_state.wager_amount = 100

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

def display_market_context():
    """
    Displays macro market context (VIX, Yields, BTC Volume) using yfinance.
    """
    try:
        # Fetch data for VIX, 10Y Yield, and BTC
        # Use simple Ticker objects for reliability
        vix = yf.Ticker("^VIX").history(period="1d")
        tnx = yf.Ticker("^TNX").history(period="1d")
        btc = yf.Ticker("BTC-USD").history(period="1d")
        
        if not vix.empty and not tnx.empty and not btc.empty:
            # VIX
            vix_val = vix['Close'].iloc[-1]
            vix_open = vix['Open'].iloc[-1]
            vix_delta = vix_val - vix_open
            
            # 10Y Yield
            tnx_val = tnx['Close'].iloc[-1]
            
            # BTC Volume
            btc_vol = btc['Volume'].iloc[-1]
            btc_vol_b = btc_vol / 1_000_000_000
            
            # Regime Logic
            if vix_val > 20:
                regime = "HIGH VOLATILITY"
                regime_icon = "⚠️"
                is_high_vol = True
            else:
                regime = "NORMAL"
                regime_icon = "✅"
                is_high_vol = False
                
            # Display
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "🌪️ VIX", 
                f"{vix_val:.2f}", 
                f"{vix_delta:+.2f}", 
                delta_color="inverse",
                help="Fear Index: <15 = Calm, 15-20 = Normal, >20 = High Fear. Rising VIX = Market Uncertainty."
            )
            m2.metric(
                "🏦 10Y Yield", 
                f"{tnx_val:.2f}%",
                help="Interest Rate Proxy: Rising yields = Expensive borrowing, often bearish for growth stocks. Falling = Cheap money."
            )
            m3.metric(
                "₿ Volume", 
                f"{btc_vol_b:.1f}B",
                help="Bitcoin Volume: High volume = Strong crypto activity. Low volume = Sleepy markets, lower conviction."
            )
            
            with m4:
                if is_high_vol:
                    st.error(f"{regime_icon} {regime}")
                else:
                    st.success(f"{regime_icon} {regime}")

            
    except Exception as e:
        # Fail silently or show small error
        st.caption(f"Market Context Unavailable: {e}")

def categorize_markets(markets, ticker):
    """
    Categorizes markets into Hourly, Daily, and Range buckets based on expiration and title.
    """
    buckets = {'hourly': [], 'daily': [], 'range': []}
    now = datetime.utcnow()
    
    # Define Timezone for "9am to 12am" logic (ET)
    # UTC-5 for simplicity (Standard Time)
    offset = timedelta(hours=-5)
    now_et = now + offset
    
    is_crypto = ticker in ["BTC", "ETH"]
    
    for m in markets:
        try:
            # Parse expiration
            exp_str = m.get('expiration')
            if not exp_str:
                continue
            
            # Handle ISO format (sometimes with Z, sometimes without)
            exp_time = pd.to_datetime(exp_str).replace(tzinfo=None)
            
            # Check for Range
            title = m.get('title', '').lower()
            if 'range' in title or 'between' in title:
                buckets['range'].append(m)
                continue
            
            # Check for Hourly (Expires within 90 mins)
            time_diff = (exp_time - now).total_seconds() / 60
            if 0 < time_diff <= 90:
                # Crypto Logic: Only 9am - 12am ET (09:00 to 23:59)
                if is_crypto:
                    if 9 <= now_et.hour <= 23:
                        buckets['hourly'].append(m)
                else:
                    # Indices: Standard logic (Accept all available)
                    buckets['hourly'].append(m)
                continue
                
            # Check for Daily Close (15:50 - 17:10 ET)
            # Convert to ET (UTC-5 or UTC-4). Let's approximate UTC-5 for now or use pytz if available.
            # 16:00 ET is 21:00 UTC (Standard) or 20:00 UTC (Daylight).
            # Let's just check if it expires today and is late in the day (after 20:00 UTC)
            if exp_time.date() == now.date() and exp_time.hour >= 20:
                buckets['daily'].append(m)
                continue
                
        except Exception as e:
            print(f"Error categorizing market: {e}")
            
    return buckets

def run_scanner(timeframe_override=None):
    """
    Runs the market scanner and updates session state.
    """
    tickers_to_scan = ["SPX", "Nasdaq", "BTC", "ETH"]
    all_strikes = []
    all_ranges = []
    
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers_to_scan):
        try:
            # 1. Fetch ALL Kalshi Markets first
            real_markets = get_real_kalshi_markets(ticker)
            buckets = categorize_markets(real_markets, ticker)
            
            # 2. Load Models
            # Hourly Model
            df_hourly = fetch_data(ticker=ticker, period="5d", interval="1m")
            model_hourly = load_model(ticker=ticker)
            
            # Daily Model
            df_daily = fetch_data(ticker=ticker, period="60d", interval="1h")
            model_daily = load_daily_model(ticker=ticker)
            
            # 3. Process Hourly Bucket
            if not df_hourly.empty and model_hourly and buckets['hourly']:
                df_features_hourly = create_features(df_hourly)
                pred_hourly = predict_next_hour(model_hourly, df_features_hourly, ticker=ticker)
                rmse_hourly = get_recent_rmse(model_hourly, df_hourly, ticker=ticker)
                curr_price_hourly = df_hourly['Close'].iloc[-1]
                
                # Generate signals for these specific markets
                # We need to map the markets to the signal format
                for m in buckets['hourly']:
                    strike = m.get('strike_price')
                    if not strike: continue
                    
                    # Calculate Prob
                    prob_val = calculate_probability(pred_hourly, strike, rmse_hourly)
                    
                    # Determine Action
                    # If Prob > 50 -> Buy YES
                    # If Prob < 50 -> Buy NO
                    if prob_val > 50:
                        action = "BUY YES"
                        conf = prob_val
                    else:
                        action = "BUY NO"
                        conf = 100 - prob_val
                        
                    s = {
                        'Asset': ticker,
                        'Strike': f"> ${strike}",
                        'Prob': f"{conf:.1f}%",
                        'Numeric_Prob': conf,
                        'Action': action,
                        'Timeframe': "Hourly",
                        'Date': pd.to_datetime(m['expiration']).strftime("%b %d"),
                        'Time': pd.to_datetime(m['expiration']).strftime("%I:%M %p"),
                        'RMSE': rmse_hourly,
                        'Real_Yes_Bid': m.get('yes_bid', 0),
                        'Real_No_Bid': m.get('no_bid', 0),
                        'Real_Yes_Ask': m.get('yes_ask', 0),
                        'Real_No_Ask': m.get('no_ask', 0),
                        'Has_Real_Data': True
                    }
                    
                    # Calculate Edge
                    cost = s['Real_Yes_Bid'] if "BUY YES" in action else s['Real_No_Bid']
                    s['Real_Edge'] = conf - cost
                    
                    all_strikes.append(s)

            # 4. Process Daily Bucket
            if not df_daily.empty and model_daily and buckets['daily']:
                df_features_daily, _ = prepare_daily_data(df_daily)
                pred_daily = predict_daily_close(model_daily, df_features_daily.iloc[[-1]])
                rmse_daily = df_daily['Close'].iloc[-1] * 0.01
                
                for m in buckets['daily']:
                    strike = m.get('strike_price')
                    if not strike: continue
                    
                    prob_val = calculate_probability(pred_daily, strike, rmse_daily)
                    
                    if prob_val > 50:
                        action = "BUY YES"
                        conf = prob_val
                    else:
                        action = "BUY NO"
                        conf = 100 - prob_val
                        
                    s = {
                        'Asset': ticker,
                        'Strike': f"> ${strike}",
                        'Prob': f"{conf:.1f}%",
                        'Numeric_Prob': conf,
                        'Action': action,
                        'Timeframe': "Daily",
                        'Date': pd.to_datetime(m['expiration']).strftime("%b %d"),
                        'Time': pd.to_datetime(m['expiration']).strftime("%I:%M %p"),
                        'RMSE': rmse_daily,
                        'Real_Yes_Bid': m.get('yes_bid', 0),
                        'Real_No_Bid': m.get('no_bid', 0),
                        'Real_Yes_Ask': m.get('yes_ask', 0),
                        'Real_No_Ask': m.get('no_ask', 0),
                        'Has_Real_Data': True
                    }
                    
                    cost = s['Real_Yes_Bid'] if "BUY YES" in action else s['Real_No_Bid']
                    s['Real_Edge'] = conf - cost
                    
                    all_strikes.append(s)
            
            # 5. Process Range Bucket (Simplified)
            for m in buckets['range']:
                # Just add them for display, maybe without prediction for now or simple logic
                r = {
                    'Asset': ticker,
                    'Range': m.get('title'),
                    'Predicted In Range?': "N/A", # Need range model
                    'Action': "Watch",
                    'Timeframe': "Hourly", # Usually ranges are hourly/daily
                    'Date': pd.to_datetime(m['expiration']).strftime("%b %d"),
                    'Time': pd.to_datetime(m['expiration']).strftime("%I:%M %p")
                }
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

# Market Context Bar
display_market_context()

# Debug section removed as per request


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

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 PnL Calculator")
st.session_state.wager_amount = st.sidebar.number_input(
    "Wager Amount ($)", 
    min_value=10, 
    max_value=10000, 
    value=st.session_state.wager_amount,
    step=50,
    help="How much you want to bet per trade"
)

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

st.sidebar.markdown("---")

with st.sidebar.expander("📘 Help & Strategy"):
    st.markdown("""
    **The Strategy**
    This model predicts the probability of hourly price moves. We buy when Model Confidence > Market Price.
    
    **The PnL**
    Enter your wager size above. The cards calculate your profit if the price hits the strike.
    
    **Bid vs Ask**
    You pay the 'Ask' price to enter a trade. If you want to sell early, you sell at the 'Bid'.
    """)

st.sidebar.markdown("---")

# DEBUG TOOLS
with st.sidebar.expander("🔧 Dev Tools"):
    st.caption("Kalshi API Debug Information")
    
    # Test one ticker in detail
    st.markdown("**Quick Test: BTC**")
    try:
        import requests
        params = {"limit": 10, "status": "open"}
        headers = {}
        api_key = os.getenv("KALSHI_API_KEY")
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            st.caption("✅ Using API Key")
        else:
            st.caption("⚠️ No API Key")
            
        url = "https://api.elections.kalshi.com/trade-api/v2/markets"
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
            st.write(f"**Total Markets:** {len(markets)}")
            
            if markets:
                st.write("**First 2 markets:**")
                for m in markets[:2]:
                    st.json(m)
        else:
            st.error(f"Error: {response.text[:200]}")
    except Exception as e:
        st.error(f"Exception: {str(e)}")
    
    st.markdown("---")
    
    for ticker in ["SPX", "Nasdaq", "BTC", "ETH"]:
        markets = get_real_kalshi_markets(ticker)
        st.markdown(f"**{ticker}:** {len(markets)} markets")
        if markets:
            for m in markets[:2]:
                st.caption(f"Strike: {m.get('strike_price')} | Yes: {m.get('yes_bid')}¢")

st.markdown("---")

# --- MAIN CONTENT AREA ---
all_strikes = st.session_state.scan_results['strikes']
all_ranges = st.session_state.scan_results['ranges']

# Filter strikes for selected asset
asset_strikes_board = [s for s in all_strikes if s['Asset'] == selected_ticker]
asset_ranges_board = [r for r in all_ranges if r['Asset'] == selected_ticker]

# Fetch current price for context
try:
    df_curr = fetch_data(ticker=selected_ticker, period="1d", interval="1m")
    if not df_curr.empty:
        curr_alpha = df_curr['Close'].iloc[-1]
    else:
        curr_alpha = 0
except:
    curr_alpha = 0

# Legend Tile
with st.container(border=True):
    l1, l2, l3, l4 = st.columns(4)
    l1.caption("🟢 BUY / 🔴 SELL")
    l1.markdown("**Buy at Ask / Sell at Bid**")
    l2.caption("🎯 ITM (In The Money)")
    l2.markdown("**Winning (Price > Strike)**")
    l3.caption("📉 OTM (Out The Money)")
    l3.markdown("**Chasing (Price < Strike)**")
    l4.caption("🔥 Edge")
    l4.markdown("**Model Prob - Market Price**")

# === MASTER-DETAIL SPLIT LAYOUT ===
col_feed, col_analysis = st.columns([1, 2], gap="medium")

# LEFT COLUMN: Trade Opportunity Board
with col_feed:
    st.markdown("### 📋 Trade Opportunity Board")
    
    # Tabs for Sections
    tab_hourly, tab_daily, tab_ranges = st.tabs(["⚡ Hourly", "📅 End of Day", "🎯 Ranges"])

with tab_hourly:
    st.caption("Short-term opportunities expiring in < 90 mins")
    hourly_ops = [s for s in asset_strikes_board if s['Timeframe'] == "Hourly"]
    
    if hourly_ops:
        # Sort by Edge (Confidence)
        hourly_ops.sort(key=lambda x: x.get('Real_Edge', abs(x['Numeric_Prob'] - 50)), reverse=True)
        
        # Limit to Top 5
        hourly_ops = hourly_ops[:5]
        
        for i, op in enumerate(hourly_ops):
            # Highlight top 3 Alpha Picks
            alpha_badge = "🏆 Alpha Pick" if i < 3 else ""
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
                                badge_text = "📉 OTM"
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
                    st.markdown(f"### :green[{signal}] {alpha_badge}")
                    # Sub-header: Strike
                    st.markdown(f"**{op['Strike']}**")
                    # Badge
                    st.caption(f":{badge_color}[{badge_text}]")
                    # Context
                    st.caption(context_line)
                    # Time
                    st.caption(f"Exp: {op['Time']}")
                
                with c2:
                    # Financials
                    # Bid/Ask
                    if op.get('Has_Real_Data'):
                        bid_price = op.get('Real_Yes_Bid') if is_buy_yes else op.get('Real_No_Bid')
                        ask_price_cents = op.get('Real_Yes_Ask') if is_buy_yes else op.get('Real_No_Ask')
                        
                        bid_str = f"{bid_price}¢" if bid_price else "No Liq"
                        ask_str = f"{ask_price_cents}¢" if ask_price_cents else "No Liq"
                        
                        price_str = f"Bid: {bid_str} | Ask: {ask_str}"
                    else:
                        bid_price = None
                        ask_price_cents = None
                        price_str = "Bid: No Liq | Ask: No Liq"

                    st.markdown(f"**{price_str}**")
                    st.caption("Buy at Ask / Sell at Bid")

                    # PnL Calculator
                    wager = st.session_state.get('wager_amount', 100)
                    if ask_price_cents and ask_price_cents > 0:
                        ask_price = ask_price_cents / 100.0
                        contracts = wager / ask_price
                        potential_payout = contracts * 1.00 # Binary options pay $1.00
                        profit = potential_payout - wager
                        profit_text = f"+${profit:.2f}"
                        profit_color = "green"
                        pnl_help = "If you bet $100 and win, you get your $100 back + this Profit amount. (Payout is always $1.00 per contract)."
                    else:
                        profit_text = "—"
                        profit_color = "grey"
                        pnl_help = "No liquidity to calculate PnL."
                    
                    st.markdown(f"**PnL: :{profit_color}[{profit_text}]**", help=pnl_help)
                    
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
                        st.session_state.selected_trade_index = ('hourly', i, op)
                        st.rerun()
    else:
        st.info("No Hourly opportunities found.")

with tab_daily:
    st.caption("End-of-day predictions (4:00 PM Close)")
    daily_ops = [s for s in asset_strikes_board if s['Timeframe'] == "Daily" or s['Timeframe'] == "End of Day"]
    
    if daily_ops:
        daily_ops.sort(key=lambda x: x.get('Real_Edge', abs(x['Numeric_Prob'] - 50)), reverse=True)
        
        # Limit to Top 5
        daily_ops = daily_ops[:5]
        
        for i, op in enumerate(daily_ops):
            # Highlight top 3 Alpha Picks
            alpha_badge = "🏆 Alpha Pick" if i < 3 else ""
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
                                badge_text = "📉 OTM"
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
                    st.markdown(f"### :green[{signal}] {alpha_badge}")
                    st.markdown(f"**{op['Strike']}**")
                    st.caption(f":{badge_color}[{badge_text}]")
                    st.caption(context_line)
                    st.caption(f"Exp: {op['Time']}")
                
                with c2:
                    # Bid/Ask
                    if op.get('Has_Real_Data'):
                        bid_price = op.get('Real_Yes_Bid') if is_buy_yes else op.get('Real_No_Bid')
                        ask_price_cents = op.get('Real_Yes_Ask') if is_buy_yes else op.get('Real_No_Ask')
                        
                        bid_str = f"{bid_price}¢" if bid_price else "No Liq"
                        ask_str = f"{ask_price_cents}¢" if ask_price_cents else "No Liq"
                        
                        price_str = f"Bid: {bid_str} | Ask: {ask_str}"
                    else:
                        bid_price = None
                        ask_price_cents = None
                        price_str = "Bid: No Liq | Ask: No Liq"
                    
                    st.markdown(f"**{price_str}**")
                    st.caption("Buy at Ask / Sell at Bid")

                    # PnL Calculator
                    wager = st.session_state.get('wager_amount', 100)
                    if ask_price_cents and ask_price_cents > 0:
                        ask_price = ask_price_cents / 100.0
                        contracts = wager / ask_price
                        potential_payout = contracts * 1.00 # Binary options pay $1.00
                        profit = potential_payout - wager
                        profit_text = f"+${profit:.2f}"
                        profit_color = "green"
                        pnl_help = "If you bet $100 and win, you get your $100 back + this Profit amount. (Payout is always $1.00 per contract)."
                    else:
                        profit_text = "—"
                        profit_color = "grey"
                        pnl_help = "No liquidity to calculate PnL."
                    
                    st.markdown(f"**PnL: :{profit_color}[{profit_text}]**", help=pnl_help)
                    
                    # Model Value
                    model_val = int(conf)
                    st.caption(f"Model: {model_val}¢")
                    
                    # Edge
                    edge = abs(conf - 50)
                    if op.get('Has_Real_Data'):
                        real_edge = op.get('Real_Edge', 0)
                        edge_str = f"🔥 +{real_edge:.1f}% Edge"
                    else:
                        edge_str = f"⚡ {edge:.1f}% Conf"
                        
                    st.markdown(f"**{edge_str}**")
                    
                    if st.button("Analyze", key=f"dd_d_{i}_{op['Strike']}"):
                        st.session_state.selected_trade_index = ('daily', i, op)
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

# RIGHT COLUMN: Deep Dive Analysis
with col_analysis:
    st.markdown("### 🔬 Deep Dive Analysis")
    
    if 'selected_trade_index' in st.session_state and st.session_state.selected_trade_index:
        _, _, selected_strike = st.session_state.selected_trade_index
        
        # Ensure it matches current asset
        if selected_strike['Asset'] == selected_ticker:
            st.markdown(f"**{selected_strike['Action']} at {selected_strike['Strike']}**")
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
                
                # Parse strike price
                strike_str = str(selected_strike['Strike'])
                strike_price = float(strike_str.replace('>', '').replace('$', '').replace(',', '').strip())
                
                # Calculate probability
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
                    mask = x >= strike_price
                    fig.add_trace(go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='lines',
                        fill='tozeroy',
                        name=f'Prob Above {strike_str}',
                        line=dict(color='#22c55e', width=0),
                        fillcolor='rgba(34, 197, 94, 0.3)'
                    ))
                else:
                    mask = x <= strike_price
                    fig.add_trace(go.Scatter(
                        x=x[mask], y=y[mask],
                        mode='lines',
                        fill='tozeroy',
                        name=f'Prob Below {strike_str}',
                        line=dict(color='#ef4444', width=0),
                        fillcolor='rgba(239, 68, 68, 0.3)'
                    ))
                
                # Add vertical lines
                fig.add_vline(x=pred_deep, line_dash="dash", line_color="orange", annotation_text="Predicted", annotation_position="top")
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
                col_d1.metric("Current", f"${curr_price_deep:,.2f}")
                col_d2.metric("Predicted", f"${pred_deep:,.2f}", f"{((pred_deep - curr_price_deep) / curr_price_deep * 100):+.2f}%")
                col_d3.metric("Uncertainty", f"${rmse_deep:,.2f}")
                
            except Exception as e:
                st.error(f"Unable to load analysis: {e}")
        else:
            st.info(f"👈 Selected trade is for {selected_strike['Asset']}, but you're viewing {selected_ticker}. Please switch assets or select a different trade.")
    else:
        st.info("👈 Select a trade from the board to view deep-dive analysis.")

st.markdown("---")

# Help section moved to sidebar

# === TABS REMOVED ===
# Analytics moved to pages/1_📈_Performance.py


# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and does not constitute financial advice. Trading involves risk.")
