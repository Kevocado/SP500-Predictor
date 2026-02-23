"""
Kalshi Edge Finder — v6 (On-Demand AI + Sorted + Backtesting)
4-Tab UI: Weather Arb → Macro/Fed → Quant Lab → Backtesting
Each tab has date filter + sort controls. No batch AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone, timedelta
import re
from src.data_loader import get_macro_data
from src.news_analyzer import NewsAnalyzer
from src.microstructure_engine import MicrostructureEngine
from src.predictit_engine import PredictItEngine
from huggingface_hub import hf_hub_download

# ─── AUTO-PULL MODELS FROM HF HUB ─────────────────────────────
def ensure_models_exist():
    """Milestone: Automated Sync. Pre-warms the Hugging Face cache for all active models."""
    tickers = ["SPY", "QQQ", "BTC", "ETH"]
    repo_id = "KevinSigey/Kalshi-LightGBM"
    
    for t in tickers:
        m_file = f"models/lgbm_model_{t}.pkl"
        f_file = f"models/features_{t}.pkl"
        
        for filename in [m_file, f_file]:
            try:
                # This downloads and caches it in ~/.cache/huggingface/hub/
                # ensuring ultra-fast loading for quant_engine.py
                hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as e:
                pass # Non-fatal, engines have their own fallback logic

ensure_models_exist()

with st.sidebar:
    st.markdown("---")
    # Category filter as requested
    filter_category = st.multiselect("Focus Categories", ["Weather", "Macro", "Alpha", "Niche"], default=["Weather", "Macro", "Alpha", "Niche"])

def parse_kalshi_ticker(ticker):
    """Translates a Kalshi ticker like KXHIGHNY-26FEB23-B33.5 into plain English."""
    try:
        parts = ticker.split('-')
        if len(parts) >= 2:
            base = parts[0]
            date_str = parts[1]
            strike = parts[2] if len(parts) > 2 else ""
            
            # Formats like 26FEB23
            if len(date_str) >= 7:
                month = date_str[2:5].capitalize()
                day = int(date_str[5:7])
                date_pretty = f"{month} {day}"
            else:
                date_pretty = date_str

            # Parse strikes
            strike_clean = strike.replace('B', '').replace('T', '')
            
            if base == 'KXHIGHNY': return f"NYC High Temp ≥ {strike_clean}° on {date_pretty}"
            if base == 'KXHIGHCHI': return f"Chicago High Temp ≥ {strike_clean}° on {date_pretty}"
            if base == 'KXHIGHMIA': return f"Miami High Temp ≥ {strike_clean}° on {date_pretty}"
            if base == 'KXFED': return f"Fed Rate at {strike_clean}% on {date_pretty}"
            if base == 'KXLCPIMAXYOY': return f"CPI YoY ≥ {strike_clean}% ({date_pretty})"
            if base == 'KXGDPYEAR': return f"GDP Growth ≥ {strike_clean}% ({date_pretty})"
            if base == 'KXRECSSNBER': return f"Recession ({date_pretty})"
            if base == 'KXU3MAX': return f"Unemployment ≥ {strike_clean}% ({date_pretty})"
            
            return f"{base} {strike_clean} ({date_pretty})"
    except Exception:
        pass
    return ticker

# ─── AZURE TABLE: FETCH PRE-COMPUTED OPPORTUNITIES ────────────
@st.cache_data(ttl=30)
def fetch_opportunities():
    """Reads from both Azure tables: LiveOpportunities and PaperTradingSignals."""
    try:
        from azure.data.tables import TableClient
        conn_str = os.getenv("AZURE_CONNECTION_STRING")
        if not conn_str:
            return [], [], None

        live_opps = []
        paper_opps = []
        last_update = None

        try:
            live_client = TableClient.from_connection_string(conn_str, "LiveOpportunities")
            live_entities = list(live_client.query_entities(""))
            live_entities.sort(key=lambda x: float(x.get('Edge', 0)), reverse=True)
            live_opps = live_entities
        except Exception:
            pass

        try:
            paper_client = TableClient.from_connection_string(conn_str, "PaperTradingSignals")
            paper_entities = list(paper_client.query_entities(""))
            paper_entities.sort(key=lambda x: float(x.get('Edge', 0)), reverse=True)
            paper_opps = paper_entities
        except Exception:
            pass

        # Fallback: old table
        if not live_opps and not paper_opps:
            try:
                old_client = TableClient.from_connection_string(conn_str, "CurrentOpportunities")
                entities = list(old_client.query_entities(""))
                entities.sort(key=lambda x: float(x.get('Edge', 0)), reverse=True)
                for e in entities:
                    engine = e.get('Engine', '').lower()
                    if engine in ('weather', 'macro'):
                        live_opps.append(e)
                    else:
                        paper_opps.append(e)
            except Exception:
                pass

        all_entities = live_opps + paper_opps
        if all_entities:
            ts = all_entities[0].get('_metadata', {}).get('timestamp') or all_entities[0].get('Timestamp')
            if ts:
                if isinstance(ts, str):
                    last_update = pd.to_datetime(ts).strftime("%H:%M UTC")
                else:
                    last_update = ts.strftime("%H:%M UTC")
            else:
                last_update = datetime.now(timezone.utc).strftime("%H:%M UTC")

        return live_opps, paper_opps, last_update
    except Exception as e:
        print(f"fetch_opportunities error: {e}")
        return [], [], None


def run_ai_validation(sig):
    """Run on-demand AI validation for a single trade."""
    try:
        from src.ai_validator import AIValidator
        validator = AIValidator()
        opp = {
            'engine': sig.get('Engine', 'Unknown'),
            'asset': sig.get('Asset', ''),
            'market_title': sig.get('Market', ''),
            'action': sig.get('Action', ''),
            'edge': float(sig.get('Edge', 0)),
            'reasoning': sig.get('Reasoning', ''),
            'data_source': sig.get('DataSource', ''),
        }
        return validator.validate_trade(opp)
    except Exception as e:
        return {'approved': None, 'ai_reasoning': f'Error: {e}', 'confidence': 0, 'error': str(e)}


@st.cache_data(ttl=60)
def fetch_backtest_snapshots():
    """Fetch historical snapshots from Azure Blob."""
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_CONNECTION_STRING")
        if not conn_str:
            return []
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container = blob_service.get_container_client("market-snapshots")
        blobs = sorted(container.list_blobs(), key=lambda b: b.name, reverse=True)
        snapshots = []
        for blob in blobs[:50]:
            try:
                data = container.download_blob(blob.name).readall()
                snapshots.append(json.loads(data))
            except Exception:
                pass
        return snapshots
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
# SIDEBAR FILTERS & SETTINGS
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎓 Institutional Filters")
    min_edge_filter = st.slider(
        "Min Edge %", 0, 50, 5, 
        help="Difference between our model's probability and the market price. Higher = better margin of safety."
    )
    with st.expander("⚙️ Advanced Filters (Institutional)"):
        min_kelly_filter = st.slider(
            "Min Kelly Bet %", 0.0, 10.0, 1.0,
            help="Recommended bankroll % to wager based on the Kelly Criterion. Filters out small/low-conviction bets."
        )
        max_spread = st.slider(
            "Max Bid-Ask Spread (¢)", 1, 25, 10,
            help="Filters out illiquid markets where the difference between BUY and SELL prices is too large (wiping out profit)."
        )
        
        st.markdown("---")
        st.markdown("### 📈 Risk Management")
        prob_gate = st.checkbox(
            "Confidence Gate (15-85% Prob)", value=True,
            help="Focuses on moderate-probability events where price movements are most dynamic. Filters out 'sure things' and 'long shots'."
        )
        annualized_sort = st.checkbox(
            "Prioritize Annualized EV", value=False,
            help="Sort by Annualized Expected Value: (Edge / Days to Expiry) * 365. Helps compare short-term vs long-term trades."
        )
    
    st.markdown("---")
    st.caption("v7.0 PhD Edition")

def calculate_annualized_ev(edge_pct, expiration_str):
    """Calculates Annualized Expected Value."""
    try:
        now = datetime.now(timezone.utc)
        # Use timezone-aware comparison
        if not expiration_str:
            return 0
            
        try:
            exp = pd.to_datetime(expiration_str)
            if exp.tzinfo is None:
                exp = exp.tz_localize('UTC')
        except Exception:
            return 0
            
        days_to_res = (exp - now).days
        if days_to_res <= 0: days_to_res = 1 # Resolve today
        
        # Annualized Edge = (Edge / Days) * 365
        return (edge_pct * 365) / days_to_res
    except Exception:
        return 0

def render_grid(data, key_suffix, empty_msg="No matching opportunities found."):
    """Renders high-density data grid using native st.dataframe."""
    if not data:
        st.info(empty_msg, icon="ℹ️")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Pre-processing
    numeric_cols = ['Edge', 'ModelProb', 'MarketPrice', 'Spread', 'AnnualizedEV']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Apply Probability Gate (15-85%)
    if prob_gate and 'ModelProb' in df.columns:
        df = df[(df['ModelProb'] >= 15) & (df['ModelProb'] <= 85)]

    # 2. Apply Filters
    df = df[df['Edge'] >= min_edge_filter]
    if 'Spread' in df.columns:
        df = df[df['Spread'] <= max_spread]
    
    # Kelly filter if available
    if 'KellySuggestion' in df.columns:
        df = df[df['KellySuggestion'] >= min_kelly_filter]

    if df.empty:
        st.info(empty_msg, icon="ℹ️")
        return

    # Sort
    if annualized_sort and 'AnnualizedEV' in df.columns:
        df = df.sort_values('AnnualizedEV', ascending=False)
    else:
        df = df.sort_values('Edge', ascending=False)

    # Select columns
    final_cols = ['Asset', 'Market', 'Action', 'Edge', 'AnnualizedEV', 'MarketPrice', 'ModelProb', 'Spread', 'MarketTicker']
    
    # RENDER AS TILES (Rich Alpha Tiles)
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        col = cols[i % 2]
        with col:
            ticker = row.get('MarketTicker', 'Unknown')
            # Fetch Microstructure (Whale) Info
            ms = MicrostructureEngine()
            skew_data = ms.analyze_skew(ticker)
            whale_icon = "🐳" if skew_data.get('whale_detected') else ""
            skew_val = skew_data.get('skew', 0)
            
            # Smart Entry Logic (🔥): Edge > 10% + Whale + News Alignment
            edge_val = float(row.get('Edge', 0))
            sentiment = row.get('NewsSentiment', 'Neutral')
            action = row.get('Action', '')
            
            # Action alignment: Bullish news + BUY YES OR Bearish news + BUY NO
            news_aligned = (sentiment == "Bullish" and "YES" in action) or (sentiment == "Bearish" and "NO" in action)
            
            is_smart_entry = (edge_val > 10) and skew_data.get('whale_detected') and news_aligned
            smart_icon = "🔥 SMART ENTRY" if is_smart_entry else ""
            
            # Tile UI (Refined for Clarity)
            edge_color = "#3fb950" if edge_val > 0 else "#f85149"
            edge_emoji = "📈" if edge_val > 0 else "📉"
            
            st.markdown(f"""
            <div class="quant-card" style="border-left: 5px solid {edge_color}; margin-bottom: 25px; position: relative;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="width: 75%;">
                        <strong style="font-size: 1.15rem; color: #c9d1d9;">{row.get('Market', 'Unknown')}</strong><br>
                        <div style="margin-top: 5px;">
                            <span class="stat-pill">{row.get('Asset', 'Alpha')}</span>
                            <span class="stat-pill" style="border-color: #3fb950; color: #3fb950!important;">{whale_icon} Depth Skew: {skew_val:+.0f}%</span>
                            <span class="stat-pill" style="border-color: #f85149; color: #f85149!important; font-weight: bold;">{smart_icon}</span>
                            <span style="font-size: 0.8rem; color: #8b949e; margin-left: 5px;">{sentiment if sentiment != 'Neutral' else ''}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span class="edge-positive" style="font-size: 1.5rem; color: {edge_color}!important;">{edge_emoji} {edge_val:+.1f}%</span><br>
                        <span style="color: #8b949e; font-size: 0.85rem;">EV: {float(row.get('AnnualizedEV', 0)):.0f}% Ann.</span>
                    </div>
                </div>
                <div style="margin-top: 15px; display: flex; gap: 15px; font-size: 0.95rem; color: #8b949e; border-top: 1px solid #30363d; padding-top: 12px;">
                    <span>💵 <strong>{row.get('MarketPrice', 0)}¢</strong> Price</span>
                    <span>🎯 Prob: <strong>{row.get('ModelProb', 0):.0f}%</strong></span>
                    <span>↔️ Spread: <strong>{row.get('Spread', 0)}¢</strong></span>
                </div>
                <div style="margin-top: 15px;">
                    <a href="https://kalshi.com/markets/{ticker.split('-')[0].lower()}" target="_blank" style="text-decoration: none; width: 100%;">
                        <button style="background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important; color: white !important; border: none !important; padding: 12px 0; border-radius: 8px; cursor: pointer; font-size: 1rem; width: 100%; font-weight: bold; border: 1px solid rgba(255,255,255,0.1) !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">⚡ OPEN LIVE TRADE</button>
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    return

st.set_page_config(
    page_title="Kalshi Edge Finder",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DARK THEME CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background-color: #0d1117 !important; color: #e6edf3 !important; }
    .stApp p, .stApp span, .stApp label, .stApp div, .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], .stCaption, .stCaption p { color: #e6edf3 !important; }
    [data-testid="stMetricDelta"] { color: #3fb950 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e !important; background: transparent; }
    .stTabs [aria-selected="true"] { color: #e6edf3 !important; background: #21262d !important; border-radius: 6px; }
    .stButton > button { background: linear-gradient(135deg, #388bfd 0%, #a371f7 100%) !important; color: white !important; border: none !important; font-weight: 600; }
    .stButton > button:hover { opacity: 0.85; }
    .hero-header { background: linear-gradient(135deg, rgba(56, 139, 253, 0.1) 0%, rgba(163, 113, 247, 0.1) 100%); border: 1px solid rgba(56, 139, 253, 0.2); border-radius: 16px; padding: 24px 32px; margin-bottom: 24px; }
    .hero-header h1 { background: linear-gradient(90deg, #388bfd, #a371f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; font-weight: 700; margin: 0; }
    .hero-header p { color: #8b949e !important; margin: 4px 0 0 0; }
    .quant-card { background: linear-gradient(135deg, rgba(56, 139, 253, 0.08) 0%, rgba(100, 160, 255, 0.04) 100%); border: 1px solid rgba(56, 139, 253, 0.3); border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; }
    .edge-positive { color: #3fb950 !important; font-weight: 700; }
    .edge-negative { color: #f85149 !important; font-weight: 700; }
    .stat-pill { display: inline-block; background: rgba(56, 139, 253, 0.12); border: 1px solid rgba(56, 139, 253, 0.3); border-radius: 20px; padding: 4px 12px; font-size: 0.8rem; color: #388bfd !important; margin-right: 6px; }
    .date-pill { display: inline-block; background: rgba(163, 113, 247, 0.12); border: 1px solid rgba(163, 113, 247, 0.3); border-radius: 20px; padding: 4px 12px; font-size: 0.8rem; color: #a371f7 !important; margin-right: 6px; }
    .ai-approved { background: rgba(63, 185, 80, 0.15); border: 1px solid rgba(63, 185, 80, 0.4); border-radius: 8px; padding: 10px 14px; margin-top: 8px; font-size: 0.85rem; color: #3fb950 !important; }
    .ai-rejected { background: rgba(248, 81, 73, 0.15); border: 1px solid rgba(248, 81, 73, 0.4); border-radius: 8px; padding: 10px 14px; margin-top: 8px; font-size: 0.85rem; color: #f85149 !important; }
</style>
""", unsafe_allow_html=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
<h1>⛈️ Kalshi Edge Finder</h1>
<p>Weather Arbitrage • FRED Economics • On-Demand AI Validation</p>
</div>
""", unsafe_allow_html=True)

live_opps, paper_opps, last_updated = fetch_opportunities()
live_opps = live_opps or []
paper_opps = paper_opps or []
all_entities = live_opps + paper_opps

weather_opps = [o for o in live_opps if o.get('Engine', '').lower() == 'weather']
macro_opps = [o for o in live_opps if o.get('Engine', '').lower() == 'macro']
niche_opps = [o for o in live_opps if o.get('Engine', '').lower() in ('tsa', 'eia')]

col1, col2, col3, col4 = st.columns(4)
col1.metric("⛈️ Weather", len(weather_opps))
col2.metric("🏛️ Macro", len(macro_opps))
col3.metric("🧪 Quant", len(paper_opps))
col4.markdown(f"**📅 Last Sync:** {last_updated or 'N/A'}")

if st.button("🔄 Request New Scan", use_container_width=True):
    st.toast("Scan requested. Updates arriving in ~30 seconds.", icon="⏳")
    st.cache_data.clear()
    import time
    time.sleep(1) # Visual delay for toast
    st.rerun()

# ─── MARKET HEAT GAUGE (PhD Integration) ───────────────────────────
macro_data = get_macro_data()
vix = macro_data.get('vix', 20)
yc = macro_data.get('yield_curve', 0)

# Calculate Heat Score (-100 to 100)
# VIX > 20 is 'hot' (fear), VIX < 15 is 'cool' (calm)
# Yield Curve < 0 is 'hot' (recession risk)
heat_score = ((vix - 15) * 5) - (yc * 50) 
heat_label = "FEAR" if heat_score > 30 else ("GREED" if heat_score < -30 else "NEUTRAL")
heat_color = "#f85149" if heat_score > 30 else ("#3fb950" if heat_score < -30 else "#8b949e")

st.markdown(f"""
<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px; margin-bottom: 25px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase;">Live Market Sentiment</span><br>
            <strong style="font-size: 1.8rem; color: {heat_color};">{heat_label}</strong>
        </div>
        <div style="text-align: right;">
            <span style="color: #8b949e; font-size: 0.8rem;">VIX: {vix:.2f} | 10Y-2Y: {yc:.2f}</span><br>
            <div style="width: 150px; height: 8px; background-color: #30363d; border-radius: 4px; margin-top: 5px;">
                <div style="width: {min(max(heat_score + 50, 0), 100)}%; height: 100%; background-color: {heat_color}; border-radius: 4px;"></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── 8-TAB LAYOUT ────────────────────────────────────────────────────
tab_portfolio, tab_weather, tab_macro, tab_niche, tab_quant, tab_arbitrage, tab_backtest, tab_help = st.tabs([
    "📁 My Portfolio",
    "⛈️ Weather Arb",
    "🏛️ Macro/Fed",
    "✈️ Niche Alpha",
    "🧪 Quant Lab (Paper)",
    "⚖️ Cross-Venue Arb",
    "📊 Backtesting",
    "🎓 Institutional Help"
])

# ═══════════════════════════════════════════════════════════════
# TAB 0: PORTFOLIO (READ-ONLY)
# ═══════════════════════════════════════════════════════════════
with tab_portfolio:
    st.warning("⚠️ **READ-ONLY SIMULATION**: This portfolio view is for education only. No real or simulated trades are executed from this app.")
    st.markdown("### 📁 My Kalshi Portfolio")
    st.caption("Live view of your positions, balance, and P&L.")

    try:
        from src.kalshi_portfolio import KalshiPortfolio, check_portfolio_available

        if not check_portfolio_available():
            st.warning("""
            **Portfolio not connected.** To enable, add your API Key ID to `.env`:

            ```
            KALSHI_API_KEY_ID=your-key-id-here
            ```

            **How to find your Key ID:**
            1. Go to [Kalshi Settings → API Keys](https://kalshi.com/account/api-keys)
            2. Copy the **Key ID** (not the private key — you already have that)
            3. Paste it as `KALSHI_API_KEY_ID` in your `.env` file
            4. Restart the app
            """)
        else:
            @st.cache_data(ttl=30)
            def fetch_portfolio():
                portfolio = KalshiPortfolio()
                return portfolio.get_portfolio_summary()

            summary = fetch_portfolio()

            if summary.get('error'):
                st.error(f"Portfolio error: {summary['error']}")
            else:
                # ── Balance & Stats Row ──
                b1, b2, b3, b4 = st.columns(4)
                if summary['balance'] is not None:
                    b1.metric("💰 Cash", f"${summary['balance']:,.2f}")
                b2.metric("📊 Positions", summary['total_positions'])
                b3.metric("💵 Invested", f"${summary['total_invested']:,.2f}")
                if summary['portfolio_value'] is not None:
                    unrealized = summary['market_exposure'] - summary['total_invested']
                    b4.metric("📈 Portfolio Value", f"${summary['portfolio_value']:,.2f}",
                              f"{'+'if unrealized>=0 else ''}{unrealized:,.2f}")

                # Equity Line Chart (reconstructed from settlements)
                settlements = summary.get('settlements', [])
                if settlements and summary['balance'] is not None:
                    # Sort settlements chronological (oldest to newest)
                    try:
                        sorted_settles = sorted(settlements, key=lambda x: x.get('settled_time', ''))
                        
                        # We work backwards: current balance - revenue = previous balance
                        chart_data = []
                        current_bal = summary['balance']
                        # Add current point
                        chart_data.append({"Time": datetime.now(timezone.utc), "Balance": current_bal})
                        
                        running_bal = current_bal
                        for s in reversed(sorted_settles):  # newest to oldest
                            revenue_cents = s.get('revenue', 0)
                            running_bal -= (revenue_cents / 100)
                            
                            s_time = s.get('settled_time')
                            if s_time:
                                try:
                                    # Kalshi returns ISO format strings
                                    dt = datetime.fromisoformat(s_time.replace('Z', '+00:00'))
                                    chart_data.append({"Time": dt, "Balance": running_bal})
                                except Exception:
                                    pass
                        
                        if len(chart_data) > 1:
                            # Reverse back to chronological for plotting
                            chart_data.reverse()
                            df_chart = pd.DataFrame(chart_data)
                            df_chart.set_index("Time", inplace=True)
                            
                            st.markdown("#### 📈 Portfolio Equity Curve")
                            st.line_chart(df_chart, use_container_width=True, height=200, color="#3fb950")
                    except Exception as e:
                        st.caption(f"Could not render equity chart: {e}")

                st.markdown("---")

                # ── Open Positions ──
                positions = summary['positions']
                if positions:
                    st.markdown("#### 📊 Open Positions")

                    # Build edge lookup from live opportunities
                    edge_lookup = {}
                    for opp in live_opps:
                        ticker = opp.get('MarketTicker', '')
                        if ticker:
                            edge_lookup[ticker] = {
                                'edge': float(opp.get('Edge', 0)),
                                'action': opp.get('Action', ''),
                                'engine': opp.get('Engine', ''),
                                'price': float(opp.get('MarketPrice', 0)),
                            }

                    for pos in positions:
                        raw_ticker = pos.get('ticker', 'Unknown')
                        readable_ticker = parse_kalshi_ticker(raw_ticker)
                        contracts = pos.get('position', 0)
                        cost = float(pos.get('total_traded_dollars', pos.get('total_traded', 0) / 100))
                        
                        avg_cost_cents = round((cost / contracts) * 100) if contracts > 0 else 0

                        # Check if our model has an edge on this position to get current price
                        model_info = edge_lookup.get(raw_ticker, {})
                        
                        # PRIORITY: (1) Live API price from portfolio summary, (2) Scanner price from edge_lookup
                        current_cents = pos.get('current_price') # Already in cents from kalshi_portfolio.py
                        if current_cents is None and model_info:
                            current_cents = model_info['price'] * 100

                        unrealized_pnl = 0
                        model_html = ""
                        exit_html = ""
                        if current_cents is not None:
                            unrealized_pnl = ((current_cents - avg_cost_cents) * contracts) / 100

                        if model_info:
                            model_edge = model_info['edge']
                            model_action = model_info['action']
                            edge_color = "#3fb950" if model_edge > 0 else "#f85149"
                            model_html = f'<span class="stat-pill" style="border-color: {edge_color}; color: {edge_color}!important;">Model: {model_action} ({model_edge:+.1f}%)</span>'
                            
                            # 🚨 Smart Exit Logic
                            exit_reasons = []
                            if model_edge < 2.0:
                                exit_reasons.append(f"Decaying Edge ({model_edge:.1f}%)")
                            if heat_label == "FEAR":
                                exit_reasons.append("High Volatility / FEAR")
                            if exit_reasons:
                                exit_text = " + ".join(exit_reasons)
                                exit_html = f'<div style="margin-top: 8px; padding: 6px 10px; border-radius: 6px; background: rgba(248, 81, 73, 0.15); border: 1px solid #f85149; color: #f85149; font-size: 0.85rem; display: inline-block;">⚠️ <strong>SMART EXIT ALERT:</strong> {exit_text}</div>'

                        pnl_class = "edge-positive" if unrealized_pnl >= 0 else "edge-negative"
                        pnl_sign = "+" if unrealized_pnl >= 0 else ""
                        pnl_display = f"{pnl_sign}${unrealized_pnl:.2f}" if current_cents else "N/A"
                        current_display = f"{current_cents:.0f}¢" if current_cents else "N/A"

                        # Target Price State
                        target_key = f"target_{raw_ticker}"
                        if target_key not in st.session_state:
                            st.session_state[target_key] = 0
                        
                        is_target_hit = False
                        if st.session_state[target_key] > 0 and current_cents and current_cents >= st.session_state[target_key]:
                            is_target_hit = True
                            
                        card_style = "border: 1px solid #3fb950; box-shadow: 0 0 10px rgba(63, 185, 80, 0.2);" if is_target_hit else ""

                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{readable_ticker}**")
                                st.caption(f"{raw_ticker} | {contracts} contracts")
                                if model_info:
                                    st.markdown(model_html, unsafe_allow_html=True)
                                if exit_html:
                                    st.markdown(exit_html, unsafe_allow_html=True)
                            with c2:
                                st.markdown(f"<div style='text-align: right;' class='{pnl_class}'><span style='font-size: 1.2rem; font-weight: bold;'>{pnl_display}</span></div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='text-align: right; color: #8b949e; font-size: 0.85rem;'>Avg Cost: <strong>{avg_cost_cents}¢</strong><br>Current: <strong>{current_display}</strong></div>", unsafe_allow_html=True)
                        
                        # Target tracking input
                        col_t1, col_t2 = st.columns([1, 3])
                        with col_t1:
                            new_target = st.number_input("Target Exit (¢)", min_value=0, max_value=100, value=st.session_state[target_key], key=f"input_{raw_ticker}", label_visibility="collapsed")
                            if new_target != st.session_state[target_key]:
                                st.session_state[target_key] = new_target
                                st.rerun()
                        with col_t2:
                            if is_target_hit:
                                st.success("🎯 Target Hit! Time to exit.")
                            elif st.session_state[target_key] > 0:
                                st.caption(f"Tracking... will alert when price hits {st.session_state[target_key]}¢")
                else:
                    st.info("No open positions. Your settled trades are shown below.")

                # ── Settlement History ──
                settlements = summary['settlements']
                if settlements:
                    st.markdown("---")
                    st.markdown("#### 📜 Recent Settlements")

                    pnl_total = summary['total_pnl']
                    pnl_class = "edge-positive" if pnl_total >= 0 else "edge-negative"
                    st.markdown(f'**Net P&L from settlements:** <span class="{pnl_class}">${pnl_total:+,.2f}</span>',
                                unsafe_allow_html=True)

                    with st.expander(f"View {len(settlements)} settlements"):
                        for s in settlements[:20]:
                            ticker = s.get('ticker', s.get('market_ticker', 'Unknown'))
                            revenue = s.get('revenue', 0) / 100
                            settled_at = s.get('settled_time', '')[:10] if s.get('settled_time') else ''
                            icon = "✅" if revenue > 0 else ("❌" if revenue < 0 else "➖")
                            st.markdown(f"{icon} **{ticker}** — ${revenue:+.2f} {'(' + settled_at + ')' if settled_at else ''}")

    except ImportError as e:
        st.error(f"Missing dependency: {e}. Run `pip install cryptography`.")
    except Exception as e:
        st.warning("""
        **📁 Portfolio Setup Required**

        Your RSA private key needs to be regenerated. Here's how:

        1. Go to [Kalshi → Settings → API Keys](https://kalshi.com/account/api-keys)
        2. Delete the old key and **create a new one**
        3. **Download the `.pem` file** (don't copy-paste — that corrupts the key)
        4. Move it to your project root and rename to `kalshi_private_key.pem`
        5. Add the **Key ID** to your `.env`:
           ```
           KALSHI_API_KEY_ID=your-key-id-here
           ```
        6. Refresh this page

        The rest of the app works fine — this only affects the Portfolio tab.
        """)


def sort_and_filter_opps(opps, tab_key):
    """Render sorting and date filter controls. Returns filtered list. Safe against None values."""
    if opps is None:
        return []
    if not isinstance(opps, list):
        try:
            opps = list(opps)
        except Exception:
            return []
            
    if not opps:
        return []

    filter_col, sort_col = st.columns([1, 1])

    # ── Date Filter ──
    dates_available = sorted(set(o.get('MarketDate', '') for o in opps if o.get('MarketDate', '')))
    if dates_available:
        with filter_col:
            date_options = ['All Dates'] + dates_available
            selected_date = st.selectbox("📅 Filter by date", date_options, key=f"date_{tab_key}")
            if selected_date != 'All Dates':
                opps = [o for o in opps if o.get('MarketDate', '') == selected_date]

    # ── Sort ──
    with sort_col:
        sort_by = st.selectbox("🔀 Sort by", [
            "Edge (Highest First)",
            "Edge (Lowest First)",
            "Date (Soonest First)",
            "Date (Latest First)",
            "Confidence (Highest)",
        ], key=f"sort_{tab_key}")

    if sort_by == "Edge (Highest First)":
        opps.sort(key=lambda x: float(x.get('Edge', 0)), reverse=True)
    elif sort_by == "Edge (Lowest First)":
        opps.sort(key=lambda x: float(x.get('Edge', 0)))
    elif sort_by == "Date (Soonest First)":
        opps.sort(key=lambda x: x.get('MarketDate', '') or '9999')
    elif sort_by == "Date (Latest First)":
        opps.sort(key=lambda x: x.get('MarketDate', '') or '', reverse=True)
    elif sort_by == "Confidence (Highest)":
        opps.sort(key=lambda x: float(x.get('Confidence', 0)), reverse=True)

    return opps


def render_live_card(sig, card_index):
    """Render a live opportunity card with date badge and on-demand AI."""
    engine = sig.get('Engine', 'Unknown')
    edge_val = float(sig.get('Edge', 0))
    edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
    edge_sign = "+" if edge_val > 0 else ""
    reasoning = sig.get('Reasoning', '')
    market_date = sig.get('MarketDate', '')

    # Format date nicely
    date_display = ''
    if market_date:
        try:
            dt = datetime.strptime(market_date, '%Y-%m-%d')
            today = datetime.now().date()
            if dt.date() == today:
                date_display = '📅 Today'
            elif dt.date() == today + timedelta(days=1):
                date_display = '📅 Tomorrow'
            else:
                date_display = f'📅 {dt.strftime("%b %d")}'
        except Exception:
            date_display = f'📅 {market_date}'

    date_html = f'<span class="date-pill">{date_display}</span>' if date_display else ''

    st.markdown(f"""
    <div class="quant-card">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <strong style="color: #c9d1d9; font-size: 1.05rem;">{sig.get('Market', sig.get('Asset', 'Unknown'))}</strong>
                <span class="stat-pill">{engine}</span>
                <span class="stat-pill">{sig.get('Action', 'TRADE')}</span>
                {date_html}
            </div>
            <div style="text-align: right;">
                <span class="{edge_class}" style="font-size: 1.3rem;">{edge_sign}{edge_val:.1f}%</span>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 0.85rem; color: #8b949e;">
            <p><b>Analysis:</b> {reasoning}</p>
            <p><b>Data Source:</b> {sig.get('DataSource', 'N/A')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Kalshi link & AI validate button
    link_col, ai_col = st.columns([2, 1])
    kalshi_url = sig.get('KalshiUrl', '')
    if kalshi_url:
        link_col.markdown(f"[🔗 View on Kalshi →]({kalshi_url})")

    btn_key = f"ai_validate_{card_index}"
    if ai_col.button("🤖 AI Validate", key=btn_key):
        with st.spinner("Checking with Gemini 2.5 Flash..."):
            result = run_ai_validation(sig)

        approved = result.get('approved')
        ai_text = result.get('ai_reasoning', 'No response')
        confidence = result.get('confidence', 0)
        error = result.get('error', '')

        if approved is True:
            st.markdown(f"""<div class="ai-approved">
                ✅ <b>AI APPROVED</b> (Confidence: {confidence}/10)<br>{ai_text}
            </div>""", unsafe_allow_html=True)
        elif approved is False:
            st.markdown(f"""<div class="ai-rejected">
                ❌ <b>AI REJECTED</b> (Confidence: {confidence}/10)<br>{ai_text}
            </div>""", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ AI Error: {error or ai_text}")

        risk_factors = result.get('risk_factors', [])
        if risk_factors:
            with st.expander("📋 Risk Factors"):
                for rf in risk_factors:
                    st.markdown(f"- {rf}")


# ═══════════════════════════════════════════════════════════════
# TAB 1: WEATHER ARBITRAGE
# ═══════════════════════════════════════════════════════════════
    try:
        col_w1, col_w2 = st.columns([3, 1])
        
        with col_w1:
            st.markdown("### ⛈️ Weather Arbitrage — NWS Official Data")
            st.caption("NWS is the settlement source. Expired same-day markets are auto-filtered. "
                       "Click 🤖 AI Validate for Gemini's risk analysis.")

            filtered_weather = sort_and_filter_opps(list(weather_opps), "weather")

            if "Weather" in filter_category:
                render_grid(filtered_weather, "weather", empty_msg="🌤️ No weather opportunities match your filters. Try adjusting Min Edge % or Max Spread.")
            else:
                st.info("Weather category is disabled in sidebar.")
                
        with col_w2:
            st.markdown("#### 📡 Live NWS Forecast")
            st.caption("Official data from weather.gov")
            try:
                from scripts.engines.weather_engine import WeatherEngine
                we = WeatherEngine()
                forecasts = we.get_all_forecasts()
                
                for city, dates in forecasts.items():
                    city_name = {"NYC": "New York", "CHI": "Chicago", "MIA": "Miami"}.get(city, city)
                    with st.expander(f"📍 {city_name}", expanded=True):
                        # Show today and tomorrow
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        tmrw_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                        
                        if today_str in dates:
                            st.markdown(f"**Today:** High {dates[today_str]}°F")
                        if tmrw_str in dates:
                            st.markdown(f"**Tomorrow:** High {dates[tmrw_str]}°F")
            except Exception as e:
                st.caption("Unable to load NWS forecast at this time.")
    except Exception as e:
        st.error(f"Weather Tab Error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 2: MACRO/FED
# ═══════════════════════════════════════════════════════════════
with tab_macro:
    try:
        st.markdown("### 🏛️ Macro/Fed — FRED Economic Data")
        st.caption("FRED-powered predictions. Sort by date to find near-term settlements.")

        filtered_macro = sort_and_filter_opps(list(macro_opps), "macro")

        if "Macro" in filter_category:
            render_grid(filtered_macro, "macro", empty_msg="🏛️ No macroeconomic events (Fed/CPI) match your filters.")
        else:
            st.info("Macro category is disabled in sidebar.")
    except Exception as e:
        st.error(f"Macro Tab Error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 3: NICHE ALPHA (TSA / EIA)
# ═══════════════════════════════════════════════════════════════
with tab_niche:
    try:
        st.markdown("### ✈️ Niche Alpha — TSA & Energy Storage")
        st.caption("Exploiting alpha in travel volumes and energy inventory markets.")
        
        filtered_niche = sort_and_filter_opps(list(niche_opps), "niche")
        
        if "Niche" in filter_category:
            render_grid(filtered_niche, "niche", empty_msg="✈️ No alternative data alpha (TSA/EIA) matches your filters.")
        else:
            st.info("Niche Alpha category is disabled in sidebar.")
    except Exception as e:
        st.error(f"Niche Tab Error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 4: QUANT LAB (PAPER TRADING)
# ═══════════════════════════════════════════════════════════════
with tab_quant:
    try:
        st.warning("""
        ⚠️ **PAPER TRADING ONLY - EDUCATIONAL PROJECT**

        This tab predicts SPX/Nasdaq/BTC/ETH prices using LightGBM.

        **Why this is NOT real edge:**
        - Delayed data vs HFT firms with microsecond latency
        - ~50% directional accuracy = coin flip
        - Expected ROI: 0-1% after fees

        **DO NOT BET REAL MONEY ON THESE SIGNALS.**
        """)

        st.markdown("### 🧪 SPX/BTC Algorithmic Predictions (Paper Only)")

        filtered_paper = sort_and_filter_opps(list(paper_opps), "paper")

        if "Alpha" in filter_category:
            render_grid(filtered_paper, "paper", empty_msg="🧪 No quantitative signals match your active risk filters.")
        else:
            st.info("Quant Alpha category is disabled in sidebar.")
    except Exception as e:
        st.error(f"Quant Tab Error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 5: INSTITUTIONAL GLOSSARY (HELP)
# ═══════════════════════════════════════════════════════════════
with tab_help:
    st.markdown("### 🏛️ The Institutional Quant Glossary")
    st.write("Professional trading involves specific terminology. This page explains every term used in this platform.")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 📈 Alpha & Math")
        with st.expander("⭐ Edge %", expanded=True):
            st.write("""
            **What it is:** The difference between our model's predicted probability and the market's current price.
            - *Example*: If our model says a 70% chance of 'YES' and the price is 50¢ (50%), we have a **20% Edge**.
            - *Significance*: This is your profit margin. We generally only trade when Edge > 5%.
            """)
            
        with st.expander("⏳ Annualized EV", expanded=True):
            st.write("""
            **What it is:** Expected Value projected over a 365-day period.
            - *Formula*: `(Edge / Days to Expiration) * 365`.
            - *Significance*: Helps you compare a small profit that resolves today vs. a large profit that takes months. It prioritizes capital turnover.
            """)
            
        with st.expander("⚖️ Kelly Criterion", expanded=True):
            st.write("""
            **What it is:** A mathematical formula for optimal bet sizing.
            - *Significance*: It tells you exactly what % of your bankroll to risk. It balances the desire to grow your account with the need to avoid 'going broke' from a single bad trade.
            """)

    with col_b:
        st.markdown("#### 🛡️ Risk & Execution")
        with st.expander("↔️ Bid-Ask Spread", expanded=True):
            st.write("""
            **What it is:** The gap between the best 'BUY' price (Ask) and the best 'SELL' price (Bid).
            - *The Trap*: If you buy at 60¢ and the bid is 40¢, you are instantly down 20¢. 
            - *Our Rule*: We highlight markets with spreads < 5¢ to ensure you can exit the trade profitably.
            """)
            
        with st.expander("🚧 Confidence Gate (15-85%)", expanded=True):
            st.write("""
            **What it is:** A filter that hides extreme outliers.
            - *Why*: Prediction markets are most 'efficient' (and profitable) when there is actual debate. When a market is at 99%, there is no remaining profit; when it's at 1%, it's usually a lottery ticket.
            """)
            
        with st.expander("🧪 Paper Trading", expanded=True):
            st.write("""
            **What it is:** Trading with virtual money for educational and testing purposes.
            - *Note*: Our SPX/BTC models are currently in 'Paper Only' mode to prevent real-money losses while we tune the LightGBM architecture.
            """)

    st.markdown("---")
    st.caption("PhD Level Quantitative Infrastructure • Developed for Institutional Performance")

# ═══════════════════════════════════════════════════════════════
# TAB: ARBITRAGE
# ═══════════════════════════════════════════════════════════════
with tab_arbitrage:
    try:
        st.markdown("### ⚖️ Cross-Venue Arbitrage (Kalshi vs PredictIt)")
        st.markdown("""
        This lab monitors for price discrepancies between Kalshi (cents) and PredictIt ($/cents).
        *   **Arbitrage**: Delta > 8% (High Profit Opportunity)
        *   **Alignment**: Delta < 3% (Market Consensus)
        """)
        
        if st.button("🔍 Scan for PredictIt Discrepancies", key="scan_arb_pi"):
            with st.spinner("Analyzing multi-venue flow..."):
                pi_engine = PredictItEngine()
                # Ensure all_entities is defined (already handled at top of script)
                alerts = pi_engine.get_arbitrage_alerts(all_entities)
                
                if not alerts:
                    st.info("No significant price discrepancies found across major pairs.")
                else:
                    for a in alerts:
                        atype = a.get('Type', 'Alignment')
                        color = "#3fb950" if atype == "Arbitrage" else "#8b949e"
                        
                        st.markdown(f"""
                        <div class="quant-card" style="border-left: 5px solid {color}; margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <strong style="color: #c9d1d9;">{a['MarketTicker']}</strong><br>
                                    <span style="font-size: 0.85rem; color: #8b949e;">{a['PredictIt_Market']} ({a['PredictIt_Contract']})</span>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-size: 1.25rem; font-weight: bold; color: {color};">{a['Delta']:+.1f}% Delta</span><br>
                                    <span class="stat-pill">{atype}</span>
                                </div>
                            </div>
                            <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.95rem;">
                                <span>🏛️ Kalshi: <strong>{a['Kalshi_Price']:.0f}¢</strong></span>
                                <span>⚖️ PredictIt: <strong>{a['PI_Price']:.0f}¢</strong></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Arbitrage Tab Error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 4: BACKTESTING
# ═══════════════════════════════════════════════════════════════
with tab_backtest:
    try:
        st.markdown("### 📊 Backtesting — Engine Performance")

        bt_weather, bt_macro, bt_quant = st.tabs([
            "⛈️ Weather", "🏛️ Macro/Fed", "🧪 Quant ML"
        ])

        with bt_weather:
            st.markdown("#### ⛈️ NWS Temperature Prediction Accuracy")
            snapshots = fetch_backtest_snapshots()
            weather_records = [{'timestamp': s.get('timestamp_utc', ''),
                                'live_opps': s.get('live_opportunities', 0),
                                'total': s.get('markets_analyzed', 0)} for s in snapshots]
            if weather_records:
                df = pd.DataFrame(weather_records)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values('timestamp')
                if len(df) > 1:
                    st.line_chart(df.set_index('timestamp')['live_opps'], use_container_width=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Scans", len(df))
                    c2.metric("Avg Opps", f"{df['live_opps'].mean():.0f}")
                    c3.metric("Peak", int(df['live_opps'].max()))
                else:
                    st.info("Need more snapshots. Run scanner a few more times.")
            else:
                st.info("No snapshots yet. Run the background scanner.")
    except Exception as e:
        st.error(f"Backtesting Tab Error: {e}")

    with bt_macro:
        st.markdown("#### 🏛️ FRED Economic History")
        try:
            import fredapi
            from dotenv import load_dotenv
            load_dotenv()
            fred_key = os.getenv('FRED_API_KEY')
            if fred_key:
                fred = fredapi.Fred(api_key=fred_key)
                cpi = fred.get_series('CPIAUCSL', observation_start='2023-01-01')
                if len(cpi) >= 13:
                    cpi_yoy = ((cpi / cpi.shift(12)) - 1) * 100
                    cpi_yoy = cpi_yoy.dropna()
                    st.markdown("**CPI Year-over-Year (%)**")
                    st.line_chart(cpi_yoy, use_container_width=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Current CPI YoY", f"{cpi_yoy.iloc[-1]:.2f}%")
                    c2.metric("12M Range", f"{cpi_yoy.iloc[-12:].min():.1f}% — {cpi_yoy.iloc[-12:].max():.1f}%")

                fed = fred.get_series('DFEDTARU', observation_start='2023-01-01')
                if len(fed) > 0:
                    st.markdown("**Fed Funds Rate (%)**")
                    st.line_chart(fed, use_container_width=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Current Rate", f"{fed.iloc[-1]:.2f}%")
                    c2.metric("1Y Change", f"{fed.iloc[-1] - fed.iloc[min(len(fed)-1, 252)]:.2f}%")
            else:
                st.warning("FRED_API_KEY not set.")
        except Exception as e:
            st.error(f"FRED error: {e}")

    with bt_quant:
        st.markdown("#### 🧪 Quant ML — P&L Replay")
        try:
            from src.backtester import fetch_historical_data, simulate_backtest
            with st.spinner("Loading logs..."):
                logs_df = fetch_historical_data()
            if not logs_df.empty:
                c1, c2 = st.columns(2)
                bankroll = c1.number_input("Bankroll ($)", value=100, min_value=10, max_value=10000)
                min_edge = c2.slider("Min Edge (%)", 5, 30, 10)
                result = simulate_backtest(logs_df, bankroll=bankroll, min_edge=min_edge)
                m = result['metrics']
                a = result['accuracy']
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Trades", m['total_trades'])
                m2.metric("Win Rate", f"{m['win_rate']}%")
                m3.metric("Return", f"{m['total_return']}%")
                m4.metric("Sharpe", m['sharpe'])
                m5.metric("Max DD", f"-{m['max_drawdown']}%")
                if result['equity_curve']:
                    eq = pd.DataFrame(result['equity_curve'], columns=['ts', 'eq'])
                    eq['ts'] = pd.to_datetime(eq['ts'], errors='coerce')
                    eq = eq.dropna(subset=['ts'])
                    if not eq.empty:
                        st.line_chart(eq.set_index('ts')['eq'], use_container_width=True)
                if result['trades']:
                    with st.expander(f"📋 Trade Log ({len(result['trades'])} trades)"):
                        st.dataframe(pd.DataFrame(result['trades']), use_container_width=True)
            else:
                st.info("No logs yet. Run scanner a few times.")
        except Exception as e:
            st.error(f"Backtest error: {e}")
