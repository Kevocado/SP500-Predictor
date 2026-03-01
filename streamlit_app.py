"""
Kalshi Edge Finder — v7 (4-Tab Architecture)
Tabs: My Portfolio → Quant Lab → Weather Markets → Macro Markets
Execution: Human-in-the-Loop via Telegram (Kalshi API is read-only)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()


# ─── AUTO-PULL MODELS FROM HF HUB ─────────────────────────────
def ensure_models_exist():
    """Milestone: Automated Sync. Pre-warms the Hugging Face cache for all active models."""
    REPO_ID = "Kevocado/sp500-predictor-models"
    MODEL_FILES = [
        "lgbm_model_SPX.pkl", "features_SPX.pkl",
        "lgbm_model_Nasdaq.pkl", "features_Nasdaq.pkl",
    ]
    for f in MODEL_FILES:
        try:
            hf_hub_download(repo_id=REPO_ID, filename=f, cache_dir="model", force_filename=f)
        except Exception:
            pass

ensure_models_exist()


# ─── HELPER FUNCTIONS ──────────────────────────────────────────

def parse_kalshi_ticker(ticker):
    """Translates a Kalshi ticker into plain English."""
    import re
    city_map = {"NY": "NYC", "CHI": "Chicago", "MIA": "Miami"}
    m = re.match(r'KX(\w+?)(NY|CHI|MIA)-(\d{2})([A-Z]{3})(\d{2})-([AB])([\d.]+)', ticker)
    if m:
        metric, city_code, day, mon, yr, direction, strike = m.groups()
        city = city_map.get(city_code, city_code)
        dir_text = "above" if direction == "A" else "below"
        return f"{city} daily high {dir_text} {strike}°F ({day} {mon})"
    m2 = re.match(r'(KX\w+?)-([\dA-Z]+)', ticker)
    if m2:
        series, contract = m2.groups()
        series_map = {
            "KXLCPIMAXYOY": "CPI Max YoY", "KXFED": "Fed Rate",
            "KXGDPYEAR": "GDP", "KXRECSSNBER": "Recession",
            "KXFEDDECISION": "Fed Decision", "KXU3MAX": "Unemployment",
        }
        name = series_map.get(series, series)
        return f"{name}: {contract}"
    return ticker


# ─── DATA FETCHING ─────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_opportunities():
    """Reads from Supabase (primary) or Azure Tables (fallback)."""
    live_opps, paper_opps, last_update = [], [], None

    # Try Supabase first
    try:
        from src.supabase_client import get_latest_opportunities
        rows = get_latest_opportunities(limit=100)
        if rows:
            for r in rows:
                entry = {
                    'Engine': r.get('engine', ''),
                    'Asset': r.get('asset', ''),
                    'Market': r.get('market_title', ''),
                    'MarketTicker': r.get('market_ticker', ''),
                    'EventTicker': r.get('event_ticker', ''),
                    'Action': r.get('action', ''),
                    'ModelProb': r.get('model_prob', 0),
                    'MarketPrice': r.get('market_price', 0),
                    'Edge': r.get('edge', 0),
                    'Reasoning': r.get('reasoning', ''),
                    'DataSource': r.get('data_source', ''),
                    'KalshiURL': r.get('kalshi_url', ''),
                    'MarketDate': r.get('market_date', ''),
                    'Expiration': r.get('expiration', ''),
                }
                eng = entry['Engine'].lower()
                if eng in ('weather', 'macro', 'tsa', 'eia'):
                    live_opps.append(entry)
                else:
                    paper_opps.append(entry)
            last_update = datetime.now(timezone.utc).strftime("%H:%M UTC")
            return live_opps, paper_opps, last_update
    except Exception:
        pass

    # Fallback to Azure
    try:
        from azure.data.tables import TableClient
        conn_str = os.getenv("AZURE_CONNECTION_STRING", "").strip('"')
        if not conn_str:
            return [], [], None
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

        all_entities = live_opps + paper_opps
        if all_entities:
            ts = all_entities[0].get('_metadata', {}).get('timestamp') or all_entities[0].get('Timestamp')
            if ts:
                last_update = ts.strftime("%H:%M UTC") if not isinstance(ts, str) else pd.to_datetime(ts).strftime("%H:%M UTC")
            else:
                last_update = datetime.now(timezone.utc).strftime("%H:%M UTC")
        return live_opps, paper_opps, last_update
    except Exception:
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


def get_macro_data():
    """Fetch VIX and Yield Curve from external sources. Cached."""
    data = {'vix': 20, 'yield_curve': 0}
    try:
        from src.sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
        vix = sa.get_vix()
        if vix:
            data['vix'] = vix
    except Exception:
        pass
    try:
        import fredapi
        fred_key = os.getenv('FRED_API_KEY', '').strip('"')
        if fred_key:
            fred = fredapi.Fred(api_key=fred_key)
            t10y2y = fred.get_series('T10Y2Y', observation_start='2024-01-01')
            if len(t10y2y) > 0:
                data['yield_curve'] = round(t10y2y.iloc[-1], 2)
    except Exception:
        pass
    return data


def calculate_annualized_ev(edge_pct, expiration_str):
    """Calculates Annualized Expected Value."""
    try:
        exp_date = pd.to_datetime(expiration_str)
        now = datetime.now(timezone.utc)
        if exp_date.tzinfo is None:
            exp_date = exp_date.replace(tzinfo=timezone.utc)
        days_to_res = (exp_date - now).days
        if days_to_res <= 0:
            days_to_res = 1
        return (edge_pct * 365) / days_to_res
    except Exception:
        return 0


def render_grid(data, key_suffix, empty_msg="No matching opportunities found."):
    """Renders high-density data grid using native st.dataframe."""
    if not data:
        st.info(empty_msg)
        return

    rows = []
    for item in data:
        if isinstance(item, dict):
            edge = float(item.get('Edge', item.get('edge', 0)))
            market_price = float(item.get('MarketPrice', item.get('market_price', 0)))
            model_prob = float(item.get('ModelProb', item.get('model_prob', 0)))
            action = item.get('Action', item.get('action', ''))
            title = item.get('Market', item.get('market_title', ''))
            engine = item.get('Engine', item.get('engine', ''))
            reasoning = item.get('Reasoning', item.get('reasoning', ''))
            kalshi_url = item.get('KalshiURL', item.get('kalshi_url', ''))
            market_date = item.get('MarketDate', item.get('market_date', ''))

            rows.append({
                'Engine': engine,
                'Market': title[:60],
                'Action': action,
                'Edge %': round(edge, 1),
                'Model Prob': round(model_prob, 1),
                'Market ¢': round(market_price, 0),
                'Date': market_date,
            })

    if not rows:
        st.info(empty_msg)
        return

    df = pd.DataFrame(rows)
    df = df.sort_values('Edge %', ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Kalshi Edge Finder",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .regime-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .regime-bullish { background: rgba(63, 185, 80, 0.15); border: 1px solid rgba(63, 185, 80, 0.4); color: #3fb950 !important; }
    .regime-bearish { background: rgba(248, 81, 73, 0.15); border: 1px solid rgba(248, 81, 73, 0.4); color: #f85149 !important; }
    .regime-neutral { background: rgba(139, 148, 158, 0.15); border: 1px solid rgba(139, 148, 158, 0.4); color: #8b949e !important; }
    .regime-greedy { background: rgba(210, 153, 34, 0.15); border: 1px solid rgba(210, 153, 34, 0.4); color: #d29922 !important; }
    .regime-fearful { background: rgba(163, 113, 247, 0.15); border: 1px solid rgba(163, 113, 247, 0.4); color: #a371f7 !important; }
</style>
""", unsafe_allow_html=True)


# ─── HERO HEADER + AI REGIME BADGE ───────────────────────────────────

@st.cache_data(ttl=86400)
def get_ai_sentiment_cache():
    """Fetch AI market sentiment (cached 24h)."""
    snippets = []
    try:
        from scripts.engines.macro_engine import MacroEngine
        me = MacroEngine()
        cpi = me.get_latest_cpi_yoy()
        fed = me.get_fed_rate_prediction()
        gdp = me.get_gdp_prediction()
        unemp = me.get_unemployment_rate()
        if cpi is not None: snippets.append(f"CPI YoY: {cpi}%")
        if fed is not None: snippets.append(f"Fed Rate: {fed}%")
        if gdp is not None: snippets.append(f"GDP Growth: {gdp}%")
        if unemp is not None: snippets.append(f"Unemployment: {unemp}%")
    except Exception as e:
        snippets.append(f"FRED unavailable ({e})")

    if not snippets:
        snippets = ["No macro data available."]

    try:
        from src.news_analyzer import NewsAnalyzer
        analyzer = NewsAnalyzer()
        return analyzer.get_general_sentiment(vix_value=15.0, macro_news_snippets=snippets)
    except Exception:
        return {"heat_score": 0, "label": "Neutral", "summary": "Sentiment engine offline."}


sentiment = get_ai_sentiment_cache()
label = sentiment.get('label', 'Neutral')
summary = sentiment.get('summary', '')
heat = sentiment.get('heat_score', 0)

# Map label to regime name and CSS class
regime_map = {
    'Bullish': ('ACCUMULATION', 'regime-bullish'),
    'Greedy': ('EUPHORIA', 'regime-greedy'),
    'Bearish': ('DISTRIBUTION', 'regime-bearish'),
    'Fearful': ('CAPITULATION', 'regime-fearful'),
    'Neutral': ('RANGING', 'regime-neutral'),
}
regime_name, regime_css = regime_map.get(label, ('RANGING', 'regime-neutral'))

# Header with inline regime badge
hcol1, hcol2 = st.columns([3, 2])
with hcol1:
    st.markdown("""
    <div class="hero-header">
        <h1>⛈️ Kalshi Edge Finder</h1>
        <p>Weather Arbitrage • FRED Economics • ML Directional Prediction</p>
    </div>
    """, unsafe_allow_html=True)

with hcol2:
    st.markdown(f"""
    <div style="text-align: right; padding: 24px 16px 0 0;">
        <span class="regime-badge {regime_css}">🧠 AI Regime: {regime_name}</span>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Full AI Opinion"):
        st.caption(summary)


# ─── MARKET HEAT STRIP ──────────────────────────────────────────────
macro_data = get_macro_data()
vix = macro_data.get('vix', 20)
yc = macro_data.get('yield_curve', 0)
heat_score = ((vix - 15) * 5) - (yc * 50)
heat_score = max(-100, min(100, heat_score))
heat_color = "#f85149" if heat_score > 30 else ("#3fb950" if heat_score < -30 else "#d29922")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("VIX", f"{vix:.2f}")
col_m2.metric("10Y-2Y", f"{yc:.2f}")
col_m3.metric("Heat Score", f"{heat_score:.0f}")


# ─── FETCH DATA ──────────────────────────────────────────────────────
live_opps, paper_opps, last_updated = fetch_opportunities()
live_opps = live_opps or []
paper_opps = paper_opps or []

weather_opps = [o for o in live_opps if o.get('Engine', '').lower() == 'weather']
macro_opps = [o for o in live_opps if o.get('Engine', '').lower() == 'macro']

if st.button("🔄 Request New Scan", use_container_width=True):
    st.toast("Scan requested. Updates arriving in ~30 seconds.", icon="⏳")
    st.cache_data.clear()
    import time
    time.sleep(1)
    st.rerun()

st.markdown("---")

# ─── DATA COVERAGE FOOTER ────────────────────────────────────────────
try:
    from src.supabase_client import get_wipe_date
    wipe_date = get_wipe_date()
    if wipe_date:
        st.caption(f"📊 Historical Data Coverage: {wipe_date[:10]} → Present")
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════
# 4-TAB LAYOUT
# ═══════════════════════════════════════════════════════════════

tab_portfolio, tab_quant, tab_weather, tab_macro = st.tabs([
    "📁 My Portfolio",
    "🧪 Quant Lab (SPY/QQQ)",
    "⛈️ Weather Markets",
    "🏛️ Macro Markets",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: MY PORTFOLIO
# ═══════════════════════════════════════════════════════════════
with tab_portfolio:
    st.markdown("### 📁 My Kalshi Portfolio")
    try:
        from src.kalshi_portfolio import KalshiPortfolio, check_portfolio_available

        if not check_portfolio_available():
            st.warning("""
            **Portfolio Setup Required**: Add `KALSHI_API_KEY_ID` to your `.env` file.
            1. Go to [Kalshi](https://kalshi.com) → Settings → API Keys
            2. Copy the **Key ID** and paste it as `KALSHI_API_KEY_ID`
            3. Restart the app
            """)
        else:
            @st.cache_data(ttl=30)
            def fetch_portfolio():
                kp = KalshiPortfolio()
                return kp.get_portfolio_summary()

            summary = fetch_portfolio()

            if summary.get('error'):
                st.error(f"Portfolio error: {summary['error']}")
            else:
                # ── Balance & Stats Row ──
                b1, b2, b3, b4 = st.columns(4)
                if summary['balance'] is not None:
                    b1.metric("💰 Cash", f"${summary['balance']:,.2f}")
                b2.metric("📊 Positions", len(summary.get('positions', [])))
                b3.metric("📈 Settlements", len(summary.get('settlements', [])))
                unrealized = sum(
                    ((p.get('market_price', 0) - p.get('average_price', 0)) * p.get('position', 0)) / 100
                    for p in summary.get('positions', []) if p.get('market_price')
                )
                b4.metric("💹 Unrealized PnL",
                          f"{'+'if unrealized>=0 else ''}{unrealized:,.2f}")

                # ── Telegram Kill Switch ──
                st.markdown("---")
                ks_col1, ks_col2 = st.columns([3, 1])
                with ks_col1:
                    st.markdown("#### 🚨 Kill Switch")
                    st.caption("Sends Telegram alert to liquidate all positions (Kalshi API is read-only).")
                with ks_col2:
                    if st.button("🔴 Send Kill Alert", type="primary", use_container_width=True):
                        try:
                            from src.telegram_notifier import TelegramNotifier
                            tn = TelegramNotifier()
                            positions = summary.get('positions', [])
                            result = tn.alert_kill_switch(
                                reason="Manual kill switch activated from dashboard",
                                positions=positions
                            )
                            if result:
                                st.success("✅ Kill switch alert sent to Telegram!")
                            else:
                                st.error("Failed to send Telegram alert. Check bot config.")
                        except Exception as e:
                            st.error(f"Telegram error: {e}")

                # ── Open Positions ──
                st.markdown("---")
                positions = summary.get('positions', [])
                if positions:
                    st.markdown("#### 📊 Open Positions")

                    # Build market context lookup
                    context_lookup = {}
                    for opp in live_opps:
                        ticker = opp.get('MarketTicker', opp.get('market_ticker', ''))
                        if ticker:
                            context_lookup[ticker] = {
                                'edge': float(opp.get('Edge', opp.get('edge', 0))),
                                'action': opp.get('Action', opp.get('action', '')),
                                'engine': opp.get('Engine', opp.get('engine', '')),
                            }

                    for pos in positions:
                        raw_ticker = pos.get('ticker', 'Unknown')
                        readable = parse_kalshi_ticker(raw_ticker)
                        contracts = pos.get('position', 0)
                        avg_cost = pos.get('average_price', 0)
                        current = pos.get('market_price')
                        unrealized_pnl = ((current - avg_cost) * contracts) / 100 if current else 0

                        # Market Context badge
                        ctx = context_lookup.get(raw_ticker)
                        context_html = ""
                        if ctx:
                            edge_color = "#3fb950" if ctx['edge'] > 0 else "#f85149"
                            context_html = f'<span class="stat-pill" style="color:{edge_color}!important">{ctx["engine"]}: {ctx["edge"]:+.1f}% edge</span>'

                        pnl_color = "#3fb950" if unrealized_pnl >= 0 else "#f85149"
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{readable}**")
                                st.caption(f"`{raw_ticker}` | {contracts} contracts @ {avg_cost}¢")
                                if context_html:
                                    st.markdown(context_html, unsafe_allow_html=True)
                            with c2:
                                if current:
                                    st.metric("PnL", f"${unrealized_pnl:+.2f}", f"{current:.0f}¢")
                                else:
                                    st.metric("PnL", "N/A")
                else:
                    st.info("No open positions. Your settled trades are shown below.")

                # ── Settlement History ──
                settlements = summary.get('settlements', [])
                if settlements:
                    st.markdown("---")
                    st.markdown("#### 📜 Recent Settlements")
                    for s in settlements[:10]:
                        revenue = s.get('revenue', 0) / 100
                        ticker = s.get('ticker', 'Unknown')
                        settled_at = s.get('settled_time', '')[:10] if s.get('settled_time') else ''
                        icon = "✅" if revenue > 0 else ("❌" if revenue < 0 else "➖")
                        st.markdown(f"{icon} **{ticker}** — ${revenue:+.2f} {'(' + settled_at + ')' if settled_at else ''}")

    except Exception as e:
        st.warning(f"Portfolio unavailable: {e}")


# ═══════════════════════════════════════════════════════════════
# TAB 2: QUANT LAB (SPY/QQQ)
# ═══════════════════════════════════════════════════════════════
with tab_quant:
    st.markdown("### 🧪 Quant Lab — SPY/QQQ Directional Intelligence")
    st.caption("Hourly predictions using Alpaca + FinBERT + market microstructure. Models SPX/Nasdaq direction via SPY/QQQ proxies.")

    st.warning("""
    ⚠️ **PAPER TRADING ONLY**: This is a quantitative research platform. No real trades are executed.
    All signals require manual review and execution on Kalshi.
    """)

    # Display Quant signals from scanner
    if paper_opps:
        render_grid(paper_opps, "quant", empty_msg="No quant signals available.")
    else:
        st.info("🔬 No quant signals yet. Run the background scanner to generate predictions.")

    # Placeholders for Phase 3 microstructure metrics
    st.markdown("---")
    st.markdown("#### 📊 Market Microstructure (Coming Phase 3)")
    m1, m2, m3 = st.columns(3)
    m1.metric("GEX (SPY)", "—", help="Gamma Exposure from options chains")
    m2.metric("Amihud Ratio", "—", help="Illiquidity ratio: |return| / $volume")
    m3.metric("CS Spread", "—", help="Corwin-Schultz synthetic bid-ask spread")


# ═══════════════════════════════════════════════════════════════
# TAB 3: WEATHER MARKETS
# ═══════════════════════════════════════════════════════════════
with tab_weather:
    st.markdown("### ⛈️ Weather Markets — NWS Settlement Arbitrage")
    st.caption("NWS is the official settlement source for Kalshi weather markets. 6AM-6PM daily highs.")

    col_w1, col_w2 = st.columns([3, 1])

    with col_w1:
        if weather_opps:
            render_grid(weather_opps, "weather",
                        empty_msg="🌤️ No weather opportunities found.")
        else:
            st.info("🌤️ No weather opportunities. Run the scanner or wait for markets to open.")

    with col_w2:
        st.markdown("##### 📡 Live NWS Forecast")
        st.caption("Official data from weather.gov (6AM-6PM highs)")
        try:
            from scripts.engines.weather_engine import WeatherEngine
            we = WeatherEngine()
            forecasts = we.get_all_forecasts()
            for city, dates in forecasts.items():
                city_name = {"NYC": "New York", "Chicago": "Chicago", "Miami": "Miami"}.get(city, city)
                with st.expander(f"📍 {city_name}", expanded=True):
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    tmrw_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                    if today_str in dates:
                        st.markdown(f"**Today:** High {dates[today_str]}°F")
                    if tmrw_str in dates:
                        st.markdown(f"**Tomorrow:** High {dates[tmrw_str]}°F")
        except Exception:
            st.caption("Unable to load NWS forecast at this time.")

    # Telegram alert thresholds
    st.markdown("---")
    st.markdown("#### 📱 Telegram Alert Thresholds")
    st.caption("When NWS prints a temperature that guarantees a contract outcome, a Telegram alert fires automatically.")
    tc1, tc2 = st.columns(2)
    tc1.info("🌤️ **Take-Profit**: Alert when settlement is guaranteed and position held")
    tc2.info("📡 **New Edge**: Alert when edge > 15% on any weather market")


# ═══════════════════════════════════════════════════════════════
# TAB 4: MACRO MARKETS
# ═══════════════════════════════════════════════════════════════
with tab_macro:
    st.markdown("### 🏛️ Macro Markets — FRED Economic Intelligence")
    st.caption("Live economic data from the Federal Reserve Bank of St. Louis.")

    # Display macro opportunities
    if macro_opps:
        st.markdown("#### Active Macro Opportunities")
        render_grid(macro_opps, "macro", empty_msg="No macro opportunities found.")
        st.markdown("---")

    # Live FRED data
    st.markdown("#### 📈 Live Economic Indicators")
    try:
        import fredapi
        fred_key = os.getenv('FRED_API_KEY', '').strip('"')
        if fred_key:
            fred = fredapi.Fred(api_key=fred_key)

            fc1, fc2 = st.columns(2)

            with fc1:
                try:
                    cpi = fred.get_series('CPIAUCSL', observation_start='2023-01-01')
                    if len(cpi) >= 13:
                        cpi_yoy = ((cpi / cpi.shift(12)) - 1) * 100
                        cpi_yoy = cpi_yoy.dropna()
                        st.markdown("**CPI Year-over-Year (%)**")
                        st.line_chart(cpi_yoy, use_container_width=True, height=200)
                        st.metric("Current CPI YoY", f"{cpi_yoy.iloc[-1]:.2f}%")
                except Exception as e:
                    st.caption(f"CPI unavailable: {e}")

            with fc2:
                try:
                    fed = fred.get_series('DFEDTARU', observation_start='2023-01-01')
                    if len(fed) > 0:
                        st.markdown("**Fed Funds Rate (%)**")
                        st.line_chart(fed, use_container_width=True, height=200)
                        st.metric("Current Rate", f"{fed.iloc[-1]:.2f}%")
                except Exception as e:
                    st.caption(f"Fed Rate unavailable: {e}")

            st.markdown("---")
            fc3, fc4 = st.columns(2)

            with fc3:
                try:
                    unemp = fred.get_series('UNRATE', observation_start='2023-01-01')
                    if len(unemp) > 0:
                        st.markdown("**Unemployment Rate (%)**")
                        st.line_chart(unemp, use_container_width=True, height=200)
                        st.metric("Current Unemployment", f"{unemp.iloc[-1]:.1f}%")
                except Exception as e:
                    st.caption(f"Unemployment unavailable: {e}")

            with fc4:
                try:
                    gdp = fred.get_series('A191RL1Q225SBEA', observation_start='2022-01-01')
                    if len(gdp) > 0:
                        st.markdown("**GDP Growth Rate (%)**")
                        st.line_chart(gdp, use_container_width=True, height=200)
                        st.metric("Latest GDP Growth", f"{gdp.iloc[-1]:.1f}%")
                except Exception as e:
                    st.caption(f"GDP unavailable: {e}")
        else:
            st.warning("FRED_API_KEY not configured.")
    except Exception as e:
        st.error(f"FRED error: {e}")

    # PnL Backtest placeholder
    st.markdown("---")
    st.markdown("#### 📊 Model PnL Backtest")
    st.caption("Compares model rate predictions against actual Kalshi contract outcomes.")
    try:
        from src.supabase_client import get_trade_history
        trades = get_trade_history(limit=100)
        if trades:
            df_trades = pd.DataFrame(trades)
            if 'pnl_cents' in df_trades.columns and df_trades['pnl_cents'].notna().any():
                df_trades['cumulative_pnl'] = df_trades['pnl_cents'].cumsum() / 100
                st.line_chart(df_trades['cumulative_pnl'], use_container_width=True, height=200)
                total_pnl = df_trades['pnl_cents'].sum() / 100
                st.metric("Total PnL", f"${total_pnl:+.2f}")
            else:
                st.info("No PnL data yet. Run more scans to build history.")
        else:
            st.info("No trade history in Supabase yet. Run the scanner to populate.")
    except Exception:
        st.info("Trade history will populate after scanner runs begin writing to Supabase.")
