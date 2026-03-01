"""
Kalshi Edge Finder — v8 (Premium Dark Terminal)
6-Tab Layout: Portfolio → Quant Lab → Weather → Macro → Backtesting → Quant Glossary
Execution: Human-in-the-Loop via Telegram | Weather auto-sell via NWS settlement
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
    REPO_ID = "Kevocado/sp500-predictor-models"
    for f in ["lgbm_model_SPX.pkl", "features_SPX.pkl",
              "lgbm_model_Nasdaq.pkl", "features_Nasdaq.pkl"]:
        try:
            hf_hub_download(repo_id=REPO_ID, filename=f, cache_dir="model", force_filename=f)
        except Exception:
            pass

ensure_models_exist()


# ─── HELPERS ───────────────────────────────────────────────────

def parse_kalshi_ticker(ticker):
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
        return f"{series_map.get(series, series)}: {contract}"
    return ticker


# ─── DATA LAYER ────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_opportunities():
    live_opps, paper_opps, last_update = [], [], None
    try:
        from src.supabase_client import get_latest_opportunities
        rows = get_latest_opportunities(limit=100)
        if rows:
            for r in rows:
                entry = {
                    'Engine': r.get('engine', ''), 'Asset': r.get('asset', ''),
                    'Market': r.get('market_title', ''), 'MarketTicker': r.get('market_ticker', ''),
                    'EventTicker': r.get('event_ticker', ''), 'Action': r.get('action', ''),
                    'ModelProb': r.get('model_prob', 0), 'MarketPrice': r.get('market_price', 0),
                    'Edge': r.get('edge', 0), 'Reasoning': r.get('reasoning', ''),
                    'DataSource': r.get('data_source', ''), 'KalshiURL': r.get('kalshi_url', ''),
                    'MarketDate': r.get('market_date', ''), 'Expiration': r.get('expiration', ''),
                }
                if entry['Engine'].lower() in ('weather', 'macro', 'tsa', 'eia'):
                    live_opps.append(entry)
                else:
                    paper_opps.append(entry)
            last_update = datetime.now(timezone.utc).strftime("%H:%M UTC")
            return live_opps, paper_opps, last_update
    except Exception:
        pass
    # Azure fallback
    try:
        from azure.data.tables import TableClient
        conn_str = os.getenv("AZURE_CONNECTION_STRING", "").strip('"')
        if not conn_str:
            return [], [], None
        try:
            lc = TableClient.from_connection_string(conn_str, "LiveOpportunities")
            live_opps = sorted(list(lc.query_entities("")), key=lambda x: float(x.get('Edge', 0)), reverse=True)
        except Exception:
            pass
        try:
            pc = TableClient.from_connection_string(conn_str, "PaperTradingSignals")
            paper_opps = sorted(list(pc.query_entities("")), key=lambda x: float(x.get('Edge', 0)), reverse=True)
        except Exception:
            pass
        all_e = live_opps + paper_opps
        if all_e:
            ts = all_e[0].get('_metadata', {}).get('timestamp') or all_e[0].get('Timestamp')
            last_update = (ts.strftime("%H:%M UTC") if not isinstance(ts, str) else pd.to_datetime(ts).strftime("%H:%M UTC")) if ts else datetime.now(timezone.utc).strftime("%H:%M UTC")
        return live_opps, paper_opps, last_update
    except Exception:
        return [], [], None


def run_ai_validation(sig):
    try:
        from src.ai_validator import AIValidator
        v = AIValidator()
        opp = {'engine': sig.get('Engine', ''), 'asset': sig.get('Asset', ''),
               'market_title': sig.get('Market', ''), 'action': sig.get('Action', ''),
               'edge': float(sig.get('Edge', 0)), 'reasoning': sig.get('Reasoning', ''),
               'data_source': sig.get('DataSource', '')}
        return v.validate_trade(opp)
    except Exception as e:
        return {'approved': None, 'ai_reasoning': f'Error: {e}', 'confidence': 0}


@st.cache_data(ttl=86400)
def get_ai_sentiment_cache():
    snippets = []
    try:
        from scripts.engines.macro_engine import MacroEngine
        me = MacroEngine()
        for label, fn in [("CPI YoY", me.get_latest_cpi_yoy), ("Fed Rate", me.get_fed_rate_prediction),
                          ("GDP Growth", me.get_gdp_prediction), ("Unemployment", me.get_unemployment_rate)]:
            try:
                v = fn()
                if v is not None:
                    snippets.append(f"{label}: {v}%")
            except Exception:
                pass
    except Exception:
        pass
    if not snippets:
        snippets = ["No macro data available."]
    try:
        from src.news_analyzer import NewsAnalyzer
        return NewsAnalyzer().get_general_sentiment(vix_value=15.0, macro_news_snippets=snippets)
    except Exception:
        return {"heat_score": 0, "label": "Neutral", "summary": "Sentiment engine offline."}


def get_macro_data():
    data = {'vix': 20, 'yield_curve': 0}
    try:
        from src.sentiment import SentimentAnalyzer
        vix = SentimentAnalyzer().get_vix()
        if vix:
            data['vix'] = vix
    except Exception:
        pass
    try:
        import fredapi
        fk = os.getenv('FRED_API_KEY', '').strip('"')
        if fk:
            t = fredapi.Fred(api_key=fk).get_series('T10Y2Y', observation_start='2024-01-01')
            if len(t) > 0:
                data['yield_curve'] = round(t.iloc[-1], 2)
    except Exception:
        pass
    return data


def render_grid(data, key_suffix, empty_msg="No opportunities found."):
    if not data:
        st.info(empty_msg)
        return
    rows = []
    for item in data:
        if isinstance(item, dict):
            rows.append({
                'Engine': item.get('Engine', item.get('engine', '')),
                'Market': str(item.get('Market', item.get('market_title', '')))[:55],
                'Action': item.get('Action', item.get('action', '')),
                'Edge %': round(float(item.get('Edge', item.get('edge', 0))), 1),
                'Model P': round(float(item.get('ModelProb', item.get('model_prob', 0))), 1),
                'Price ¢': round(float(item.get('MarketPrice', item.get('market_price', 0))), 0),
                'Date': item.get('MarketDate', item.get('market_date', '')),
            })
    if not rows:
        st.info(empty_msg)
        return
    df = pd.DataFrame(rows).sort_values('Edge %', ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Kalshi Edge Finder", page_icon="⛈️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    .stApp { font-family: 'Inter', sans-serif; background: #0a0e17 !important; color: #c9d1d9 !important; }
    .stApp p, .stApp span, .stApp label, .stApp div, .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], .stCaption, .stCaption p { color: #c9d1d9 !important; }
    [data-testid="stMetricDelta"] { color: #3fb950 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; padding: 4px; gap: 2px; border: 1px solid #1e293b; }
    .stTabs [data-baseweb="tab"] { color: #6b7280 !important; background: transparent; font-size: 0.85rem; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { color: #e5e7eb !important; background: linear-gradient(135deg, #1e293b, #1a1f36) !important; border-radius: 8px; font-weight: 600; }

    /* ── Buttons ── */
    .stButton > button { background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important; color: white !important; border: none !important; font-weight: 600; border-radius: 8px; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); opacity: 0.95; }

    /* ── Cards ── */
    .quant-card { background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)); border: 1px solid #1e293b; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; backdrop-filter: blur(8px); }
    .quant-card:hover { border-color: #334155; }

    /* ── Accent Colors ── */
    .edge-positive { color: #34d399 !important; font-weight: 700; }
    .edge-negative { color: #f87171 !important; font-weight: 700; }

    /* ── Pills ── */
    .stat-pill { display: inline-block; background: rgba(37, 99, 235, 0.12); border: 1px solid rgba(37, 99, 235, 0.25); border-radius: 20px; padding: 3px 12px; font-size: 0.78rem; color: #60a5fa !important; margin-right: 6px; font-family: 'JetBrains Mono', monospace; }

    /* ── Regime Badges ── */
    .regime-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px; border-radius: 24px; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px; }
    .regime-bullish   { background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); color: #34d399 !important; }
    .regime-bearish   { background: rgba(248, 113, 113, 0.1); border: 1px solid rgba(248, 113, 113, 0.3); color: #f87171 !important; }
    .regime-neutral   { background: rgba(107, 114, 128, 0.15); border: 1px solid rgba(107, 114, 128, 0.35); color: #9ca3af !important; }
    .regime-greedy    { background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); color: #fbbf24 !important; }
    .regime-fearful   { background: rgba(167, 139, 250, 0.1); border: 1px solid rgba(167, 139, 250, 0.3); color: #a78bfa !important; }

    /* ── AI Opinion Panel ── */
    .ai-panel { background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8)); border: 1px solid #1e293b; border-radius: 12px; padding: 16px 20px; margin-top: 8px; font-size: 0.88rem; color: #94a3b8 !important; line-height: 1.6; }

    /* ── Hero ── */
    .hero-wrap { background: linear-gradient(135deg, rgba(37, 99, 235, 0.06) 0%, rgba(124, 58, 237, 0.06) 100%); border: 1px solid rgba(37, 99, 235, 0.15); border-radius: 16px; padding: 28px 36px; margin-bottom: 20px; }
    .hero-wrap h1 { background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .hero-wrap p { color: #6b7280 !important; margin: 6px 0 0 0; font-size: 0.88rem; }

    /* ── Metric Cards ── */
    .metric-strip { display: flex; gap: 12px; margin: 16px 0; }
    .metric-card { flex: 1; background: linear-gradient(135deg, #111827, #0f172a); border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px; text-align: center; }
    .metric-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

    /* ── Weather Auto-Sell Box ── */
    .auto-sell-box { background: linear-gradient(135deg, rgba(52, 211, 153, 0.06), rgba(16, 185, 129, 0.03)); border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 12px; padding: 16px 20px; margin: 12px 0; }
    .auto-sell-box h5 { color: #34d399 !important; margin: 0 0 8px 0; font-size: 0.95rem; }

    /* ── Data table ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HEADER: Hero + AI Regime (always visible)
# ═══════════════════════════════════════════════════════════════

sentiment = get_ai_sentiment_cache()
s_label = sentiment.get('label', 'Neutral')
s_summary = sentiment.get('summary', '')
regime_map = {
    'Bullish': ('ACCUMULATION', 'regime-bullish'),
    'Greedy': ('EUPHORIA', 'regime-greedy'),
    'Bearish': ('DISTRIBUTION', 'regime-bearish'),
    'Fearful': ('CAPITULATION', 'regime-fearful'),
    'Neutral': ('RANGING', 'regime-neutral'),
}
regime_name, regime_css = regime_map.get(s_label, ('RANGING', 'regime-neutral'))

hcol1, hcol2 = st.columns([3, 2])
with hcol1:
    st.markdown("""
    <div class="hero-wrap">
        <h1>⛈️ Kalshi Edge Finder</h1>
        <p>Weather Arbitrage • FRED Economics • ML Directional Prediction</p>
    </div>
    """, unsafe_allow_html=True)
with hcol2:
    st.markdown(f"""
    <div style="padding: 12px 0;">
        <div style="text-align: right; margin-bottom: 10px;">
            <span class="regime-badge {regime_css}">🧠 AI Regime: {regime_name}</span>
        </div>
        <div class="ai-panel">
            {s_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Market Heat Metrics ──
macro_data = get_macro_data()
vix = macro_data.get('vix', 20)
yc = macro_data.get('yield_curve', 0)
heat = max(-100, min(100, ((vix - 15) * 5) - (yc * 50)))
heat_color = "#f87171" if heat > 30 else ("#34d399" if heat < -30 else "#fbbf24")

st.markdown(f"""
<div class="metric-strip">
    <div class="metric-card">
        <div class="label">VIX</div>
        <div class="value" style="color: {'#f87171' if vix > 25 else '#34d399'}">{vix:.1f}</div>
    </div>
    <div class="metric-card">
        <div class="label">10Y-2Y Spread</div>
        <div class="value" style="color: {'#f87171' if yc < 0 else '#34d399'}">{yc:+.2f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Heat Score</div>
        <div class="value" style="color: {heat_color}">{heat:+.0f}</div>
    </div>
    <div class="metric-card">
        <div class="label">Last Sync</div>
        <div class="value" style="color: #60a5fa; font-size: 1rem;">{'—' if not (fetch_opportunities()[2]) else fetch_opportunities()[2]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Data Coverage ──
try:
    from src.supabase_client import get_wipe_date
    wd = get_wipe_date()
    if wd:
        st.caption(f"📊 Historical Data Coverage: {wd[:10]} → Present")
except Exception:
    pass

if st.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

live_opps, paper_opps, last_updated = fetch_opportunities()
live_opps = live_opps or []
paper_opps = paper_opps or []
weather_opps = [o for o in live_opps if o.get('Engine', '').lower() == 'weather']
macro_opps = [o for o in live_opps if o.get('Engine', '').lower() == 'macro']

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# 6-TAB LAYOUT
# ═══════════════════════════════════════════════════════════════

tab_port, tab_quant, tab_wx, tab_macro, tab_bt, tab_gloss = st.tabs([
    "📁 Portfolio",
    "🧪 Quant Lab",
    "⛈️ Weather",
    "🏛️ Macro",
    "📊 Backtesting",
    "📖 Quant Glossary",
])


# ═══════════════════════ TAB 1: PORTFOLIO ═════════════════════
with tab_port:
    st.markdown("### 📁 My Kalshi Portfolio")
    try:
        from src.kalshi_portfolio import KalshiPortfolio, check_portfolio_available

        if not check_portfolio_available():
            st.warning("**Setup**: Add `KALSHI_API_KEY_ID` to `.env` → [Kalshi API Keys](https://kalshi.com)")
        else:
            @st.cache_data(ttl=30)
            def fetch_portfolio():
                return KalshiPortfolio().get_portfolio_summary()

            summary = fetch_portfolio()
            if summary.get('error'):
                st.error(f"Portfolio error: {summary['error']}")
            else:
                # Balance strip
                b1, b2, b3, b4 = st.columns(4)
                if summary['balance'] is not None:
                    b1.metric("💰 Cash", f"${summary['balance']:,.2f}")
                b2.metric("📊 Positions", len(summary.get('positions', [])))
                b3.metric("📈 Settlements", len(summary.get('settlements', [])))
                unrealized = sum(
                    ((p.get('market_price', 0) - p.get('average_price', 0)) * p.get('position', 0)) / 100
                    for p in summary.get('positions', []) if p.get('market_price')
                )
                b4.metric("💹 Unrealized PnL", f"{'+'if unrealized>=0 else ''}{unrealized:,.2f}")

                # ── Weather Auto-Sell Engine Status ──
                st.markdown("---")
                st.markdown("""
                <div class="auto-sell-box">
                    <h5>🌤️ Weather Auto-Sell Engine</h5>
                    <span style="color: #94a3b8; font-size: 0.85rem;">
                        Monitors NWS for settlement-guaranteeing temperatures. When triggered, auto-sends a
                        <strong style="color:#34d399">SELL</strong> alert to Telegram to lock in max profit or protect from loss.
                        <br>• Only weather positions • Only SELL orders • Only on live NWS data changes
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # ── Open Positions with Market Context ──
                st.markdown("---")
                positions = summary.get('positions', [])
                if positions:
                    st.markdown("#### 📊 Open Positions")

                    ctx_lookup = {}
                    for opp in live_opps:
                        tk = opp.get('MarketTicker', opp.get('market_ticker', ''))
                        if tk:
                            ctx_lookup[tk] = {
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
                        pnl = ((current - avg_cost) * contracts) / 100 if current else 0

                        ctx = ctx_lookup.get(raw_ticker)
                        ctx_html = ""
                        if ctx:
                            ec = "#34d399" if ctx['edge'] > 0 else "#f87171"
                            ctx_html = f'<span class="stat-pill" style="color:{ec}!important;border-color:{ec}33">{ctx["engine"]}: {ctx["edge"]:+.1f}%</span>'

                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{readable}**")
                                st.caption(f"`{raw_ticker}` · {contracts} contracts @ {avg_cost}¢")
                                if ctx_html:
                                    st.markdown(ctx_html, unsafe_allow_html=True)
                            with c2:
                                st.metric("PnL", f"${pnl:+.2f}" if current else "N/A",
                                          f"{current:.0f}¢" if current else None)
                else:
                    st.info("No open positions.")

                # ── Settlement History ──
                settlements = summary.get('settlements', [])
                if settlements:
                    st.markdown("---")
                    st.markdown("#### 📜 Recent Settlements")
                    for s in settlements[:10]:
                        rev = s.get('revenue', 0) / 100
                        tk = s.get('ticker', '?')
                        when = s.get('settled_time', '')[:10] if s.get('settled_time') else ''
                        icon = "✅" if rev > 0 else ("❌" if rev < 0 else "➖")
                        st.markdown(f"{icon} **{tk}** — ${rev:+.2f} {'(' + when + ')' if when else ''}")
    except Exception as e:
        st.warning(f"Portfolio unavailable: {e}")


# ═══════════════════════ TAB 2: QUANT LAB ═════════════════════
with tab_quant:
    st.markdown("### 🧪 Quant Lab — SPY/QQQ Directional Intelligence")
    st.caption("Hourly predictions via Alpaca + FinBERT + market microstructure. Models SPX/Nasdaq direction using SPY/QQQ proxies.")

    st.info("⚠️ **Paper Trading Only** — Quantitative research platform. All signals require manual Kalshi execution.")

    if paper_opps:
        render_grid(paper_opps, "quant")
    else:
        st.info("🔬 No quant signals. Run the background scanner to generate predictions.")

    st.markdown("---")
    st.markdown("#### 📊 Market Microstructure Metrics")
    st.caption("Live values will populate after Phase 3 engine rebuild.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GEX (SPY)", "—", help="Net Gamma Exposure from options chains")
    m2.metric("Amihud", "—", help="Illiquidity ratio: |return| / $volume")
    m3.metric("CS Spread", "—", help="Corwin-Schultz synthetic bid-ask spread")
    m4.metric("RVOL", "—", help="Relative volume vs 20-period SMA")


# ═══════════════════════ TAB 3: WEATHER ═══════════════════════
with tab_wx:
    st.markdown("### ⛈️ Weather Markets — NWS Settlement Arbitrage")
    st.caption("NWS is the official settlement source. 6AM-6PM daily highs only.")

    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        if weather_opps:
            render_grid(weather_opps, "weather", empty_msg="🌤️ No weather edges found.")
        else:
            st.info("🌤️ Waiting for weather markets to open.")
    with col_w2:
        st.markdown("##### 📡 Live NWS")
        try:
            from scripts.engines.weather_engine import WeatherEngine
            forecasts = WeatherEngine().get_all_forecasts()
            for city, dates in forecasts.items():
                cn = {"NYC": "New York", "Chicago": "Chicago", "Miami": "Miami"}.get(city, city)
                with st.expander(f"📍 {cn}", expanded=True):
                    today = datetime.now().strftime("%Y-%m-%d")
                    tmrw = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                    if today in dates:
                        st.markdown(f"**Today:** {dates[today]}°F")
                    if tmrw in dates:
                        st.markdown(f"**Tomorrow:** {dates[tmrw]}°F")
        except Exception:
            st.caption("NWS forecast unavailable.")

    # Auto-sell thresholds
    st.markdown("---")
    st.markdown("""
    <div class="auto-sell-box">
        <h5>📱 Telegram Alert Thresholds</h5>
        <span style="color: #94a3b8; font-size: 0.85rem;">
            <strong>Take-Profit:</strong> NWS prints temperature guaranteeing contract outcome → auto SELL alert<br>
            <strong>Loss Protection:</strong> NWS forecast shifts against position → SELL alert to cut losses<br>
            <strong>New Edge:</strong> Edge > 15% on any weather market → BUY alert for manual execution
        </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════ TAB 4: MACRO ═════════════════════════
with tab_macro:
    st.markdown("### 🏛️ Macro Markets — FRED Economic Intelligence")

    if macro_opps:
        st.markdown("#### Active Opportunities")
        render_grid(macro_opps, "macro")
        st.markdown("---")

    st.markdown("#### 📈 Live Economic Indicators")
    try:
        import fredapi
        fk = os.getenv('FRED_API_KEY', '').strip('"')
        if fk:
            fred = fredapi.Fred(api_key=fk)
            fc1, fc2 = st.columns(2)
            with fc1:
                try:
                    cpi = fred.get_series('CPIAUCSL', observation_start='2023-01-01')
                    if len(cpi) >= 13:
                        cpi_yoy = (((cpi / cpi.shift(12)) - 1) * 100).dropna()
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
                    un = fred.get_series('UNRATE', observation_start='2023-01-01')
                    if len(un) > 0:
                        st.markdown("**Unemployment Rate (%)**")
                        st.line_chart(un, use_container_width=True, height=200)
                        st.metric("Unemployment", f"{un.iloc[-1]:.1f}%")
                except Exception as e:
                    st.caption(f"Unemployment unavailable: {e}")
            with fc4:
                try:
                    gdp = fred.get_series('A191RL1Q225SBEA', observation_start='2022-01-01')
                    if len(gdp) > 0:
                        st.markdown("**GDP Growth Rate (%)**")
                        st.line_chart(gdp, use_container_width=True, height=200)
                        st.metric("GDP Growth", f"{gdp.iloc[-1]:.1f}%")
                except Exception as e:
                    st.caption(f"GDP unavailable: {e}")
        else:
            st.warning("FRED_API_KEY not configured.")
    except Exception as e:
        st.error(f"FRED error: {e}")

    # PnL backtest
    st.markdown("---")
    st.markdown("#### 📊 Model PnL Backtest")
    st.caption("Compares model rate predictions against Kalshi contract outcomes.")
    try:
        from src.supabase_client import get_trade_history
        trades = get_trade_history(limit=100)
        if trades:
            df_t = pd.DataFrame(trades)
            if 'pnl_cents' in df_t.columns and df_t['pnl_cents'].notna().any():
                df_t['cumulative_pnl'] = df_t['pnl_cents'].cumsum() / 100
                st.line_chart(df_t['cumulative_pnl'], use_container_width=True, height=200)
                st.metric("Total PnL", f"${df_t['pnl_cents'].sum()/100:+.2f}")
            else:
                st.info("No PnL data yet.")
        else:
            st.info("No trade history yet. Run scanner to populate.")
    except Exception:
        st.info("Trade history populates after scanner writes to Supabase.")


# ═══════════════════════ TAB 5: BACKTESTING ═══════════════════
with tab_bt:
    st.markdown("### 📊 Backtesting — Engine Performance")

    bt_weather, bt_macro, bt_quant = st.tabs(["⛈️ Weather", "🏛️ Macro", "🧪 Quant ML"])

    with bt_weather:
        st.markdown("#### ⛈️ NWS Temperature Prediction Accuracy")
        try:
            from azure.storage.blob import BlobServiceClient
            conn_str = os.getenv("AZURE_CONNECTION_STRING", "").strip('"')
            if conn_str:
                blob_svc = BlobServiceClient.from_connection_string(conn_str, connection_timeout=10)
                container = blob_svc.get_container_client("market-snapshots")
                blobs = sorted(container.list_blobs(), key=lambda b: b.name, reverse=True)
                snapshots = []
                for blob in blobs[:50]:
                    try:
                        data = container.download_blob(blob.name).readall()
                        snapshots.append(json.loads(data))
                    except Exception:
                        pass
                records = [{'timestamp': s.get('timestamp_utc', ''),
                            'live_opps': s.get('live_opportunities', 0),
                            'total': s.get('markets_analyzed', 0)} for s in snapshots]
                if records:
                    df = pd.DataFrame(records)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
                    if len(df) > 1:
                        st.line_chart(df.set_index('timestamp')['live_opps'], use_container_width=True)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Scans", len(df))
                        c2.metric("Avg Opps", f"{df['live_opps'].mean():.0f}")
                        c3.metric("Peak", int(df['live_opps'].max()))
                    else:
                        st.info("Need more snapshots.")
                else:
                    st.info("No snapshots yet. Run scanner.")
            else:
                st.info("Azure connection not configured.")
        except Exception as e:
            st.info(f"Weather backtest: {e}")

    with bt_macro:
        st.markdown("#### 🏛️ FRED Economic History")
        try:
            import fredapi
            fk = os.getenv('FRED_API_KEY', '').strip('"')
            if fk:
                fred = fredapi.Fred(api_key=fk)
                cpi = fred.get_series('CPIAUCSL', observation_start='2023-01-01')
                if len(cpi) >= 13:
                    cpi_yoy = (((cpi / cpi.shift(12)) - 1) * 100).dropna()
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


# ═══════════════════════ TAB 6: QUANT GLOSSARY ════════════════
with tab_gloss:
    st.markdown("### 📖 The Quant Glossary")
    st.caption("Professional trading terminology used throughout this platform.")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 Alpha & Math")
        with st.expander("⭐ Edge %", expanded=True):
            st.markdown("""
            **Edge = Model Probability − Market Price**

            If our model says a weather contract has a 75% chance of settling YES
            but Kalshi prices it at 50¢ (50%), our edge is +25%. This is the core
            alpha signal driving every trade recommendation.
            """)
        with st.expander("📐 Kelly Criterion (Quarter-Kelly)"):
            st.markdown("""
            **Kelly % = (edge × probability) / (1 − probability)**

            We enforce **Quarter-Kelly (0.25×)** sizing to manage risk. If full Kelly says
            bet 20% of bankroll, we bet 5%. This dramatically reduces drawdowns while
            capturing ~75% of the theoretical growth rate.
            """)
        with st.expander("📊 Brier Score"):
            st.markdown("""
            **Brier = (1/N) × Σ(forecast − outcome)²**

            Measures probabilistic accuracy. Range: 0 (perfect) to 1 (worst).
            A score < 0.25 indicates the model outperforms naive coin-flip prediction.
            Used to detect model drift and trigger retraining.
            """)
        with st.expander("📉 Amihud Illiquidity Ratio"):
            st.markdown("""
            **Amihud = |Return| / Dollar Volume**

            Measures price impact per unit of dollar volume traded. High values
            indicate illiquid markets where large orders move prices significantly.
            Spikes signal potential liquidity cascades.
            """)

    with col_b:
        st.markdown("#### 🛡️ Risk & Execution")
        with st.expander("↔️ Bid-Ask Spread", expanded=True):
            st.markdown("""
            The difference between the best buy and sell prices. A 3¢ spread on a
            50¢ contract means you pay 51.5¢ to buy and receive 48.5¢ to sell.
            Tighter spreads = more liquid market = better execution.
            """)
        with st.expander("📊 Corwin-Schultz Spread"):
            st.markdown("""
            **Synthetic bid-ask spread estimated from daily High/Low prices.**

            Uses the statistical relationship between price range and volatility to
            estimate the effective spread without needing tick-level data. A key input
            to our microstructure feature cluster.
            """)
        with st.expander("🎯 GEX (Gamma Exposure)"):
            st.markdown("""
            **GEX = Σ(Open Interest × Gamma × Spot² × 0.01)**

            Aggregate gamma across all strikes. **Positive GEX** = dealers sell into
            rallies and buy dips (stabilizing). **Negative GEX** = dealers amplify
            moves (destabilizing). GEX flips are major regime changes.
            """)
        with st.expander("⏱️ Annualized EV"):
            st.markdown("""
            **AEV = Edge% × (365 / Days-to-Resolution)**

            Normalizes edge across different contract durations. A 5% edge on a
            1-day contract (AEV = 1,825%) is far more attractive than 20% edge
            on a 90-day contract (AEV = 81%).
            """)

    st.markdown("---")
    st.caption("Kalshi Edge Finder • Quantitative Infrastructure for Event Markets")
