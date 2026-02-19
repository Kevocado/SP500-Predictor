"""
Kalshi Quant Scanner — v2
Auto-scans on launch. Per-tab refresh. Direct Kalshi links. Trade reasoning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
from src.market_scanner import HybridScanner


# ─── AZURE TABLE: FETCH PRE-COMPUTED QUANT OPPORTUNITIES ────────────
@st.cache_data(ttl=60)  # Cache locally for 60s so we don't spam Azure
def fetch_opportunities():
    """Read high-conviction quant signals from Azure Table (written by background_scanner.py)."""
    try:
        from azure.data.tables import TableClient
        conn_str = os.getenv("AZURE_CONNECTION_STRING")
        if not conn_str:
            return [], None

        client = TableClient.from_connection_string(conn_str, "CurrentOpportunities")
        entities = list(client.query_entities("PartitionKey eq 'Live'"))

        # Sort by Edge (highest first)
        entities.sort(key=lambda x: float(x.get('Edge', 0)), reverse=True)

        # Get last-updated timestamp from Azure entity metadata
        last_update = None
        if entities:
            ts = entities[0].get('_metadata', {}).get('timestamp') or entities[0].get('Timestamp')
            if ts:
                if isinstance(ts, str):
                    last_update = pd.to_datetime(ts).strftime("%H:%M UTC")
                else:
                    last_update = ts.strftime("%H:%M UTC")
            else:
                last_update = datetime.now(timezone.utc).strftime("%H:%M UTC")

        return entities, last_update
    except Exception as e:
        print(f"fetch_opportunities error: {e}")
        return [], None

st.set_page_config(
    page_title="Kalshi Quant Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── FULL DARK THEME CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Force dark everywhere ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    /* Force white text on all labels, paragraphs, spans */
    .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    .stCaption, .stCaption p {
        color: #e6edf3 !important;
    }

    /* Metric deltas */
    [data-testid="stMetricDelta"] {
        color: #3fb950 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #e6edf3 !important;
        background: #21262d !important;
        border-radius: 6px;
    }

    /* Expander (backtest panel) */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    details {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    details summary {
        color: #e6edf3 !important;
    }
    details > div {
        background: #161b22 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #388bfd 0%, #a371f7 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }

    /* Dataframes */
    .stDataFrame, .stDataFrame div, .stDataFrame th, .stDataFrame td {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }

    /* Inputs */
    .stTextInput input, .stMultiSelect div {
        background: #21262d !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
    }

    /* Toggle */
    .stToggle label span {
        color: #e6edf3 !important;
    }

    /* Info/warning boxes */
    .stAlert {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
    }

    /* Charts */
    .stLineChart {
        background: #0d1117 !important;
    }

    /* Spinner */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #388bfd, #a371f7);
    }

    /* ── Card Styles ── */
    .hero-header {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.1) 0%, rgba(163, 113, 247, 0.1) 100%);
        border: 1px solid rgba(56, 139, 253, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .hero-header h1 {
        background: linear-gradient(90deg, #388bfd, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem; font-weight: 700; margin: 0;
    }
    .hero-header p { color: #8b949e !important; margin: 4px 0 0 0; }

    .quant-card {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.08) 0%, rgba(100, 160, 255, 0.04) 100%);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .edge-positive { color: #3fb950 !important; font-weight: 700; }
    .edge-negative { color: #f85149 !important; font-weight: 700; }

    .stat-pill {
        display: inline-block;
        background: rgba(56, 139, 253, 0.12);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #388bfd !important;
        margin-right: 6px;
    }

    .explain-box {
        background: rgba(139, 148, 158, 0.06);
        border-left: 3px solid #388bfd;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 0.85rem;
        color: #c9d1d9 !important;
    }
    .explain-box strong, .explain-box b {
        color: #e6edf3 !important;
    }

    .weather-card {
        background: linear-gradient(135deg, rgba(63, 185, 80, 0.08) 0%, rgba(63, 185, 80, 0.02) 100%);
        border: 1px solid rgba(63, 185, 80, 0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    .fred-card {
        background: linear-gradient(135deg, rgba(163, 113, 247, 0.08) 0%, rgba(163, 113, 247, 0.02) 100%);
        border: 1px solid rgba(163, 113, 247, 0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    .reasoning-box {
        background: rgba(56, 139, 253, 0.05);
        border: 1px solid rgba(56, 139, 253, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0 12px 0;
        font-size: 0.82rem;
        color: #c9d1d9 !important;
        line-height: 1.5;
    }
    .reasoning-box strong {
        color: #e6edf3 !important;
    }

    .kalshi-link {
        display: inline-block;
        background: linear-gradient(135deg, #388bfd 0%, #a371f7 100%);
        color: white !important;
        text-decoration: none;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .kalshi-link:hover { opacity: 0.85; }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #8b949e !important;
    }
    .empty-state h2 { color: #e6edf3 !important; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────
if 'scanner' not in st.session_state:
    st.session_state.scanner = HybridScanner()
if 'markets_loaded' not in st.session_state:
    st.session_state.markets_loaded = False
if 'scan_time' not in st.session_state:
    st.session_state.scan_time = None
# Per-tab data
for key in ['quant', 'weather_arb', 'fred', 'smart', 'weather_raw', 'arb', 'yield']:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── AUTO-FETCH ON FIRST LOAD ───────────────────────────────────────
# Fetches Kalshi markets + runs lightweight (non-ML) tab analyses.
# Quant tab reads from Azure Table (background_scanner.py writes to it).
if not st.session_state.markets_loaded:
    with st.spinner("📡 Loading live markets..."):
        scanner = st.session_state.scanner
        scanner.fetch_markets()
        st.session_state.markets_loaded = True
        st.session_state.scan_time = datetime.now(timezone.utc)

        # Run non-quant tab analyses (these are fast — no ML)
        st.session_state.weather_arb = scanner.scan_weather_arb()
        st.session_state.fred = scanner.scan_fred()
        st.session_state.smart = scanner.scan_smart_money()
        st.session_state.weather_raw = scanner.scan_weather_raw()
        st.session_state.arb = scanner.scan_arbitrage()
        st.session_state['yield'] = scanner.scan_yield_farms()

        # Auto-load backtest
        try:
            from src.backtester import fetch_historical_data, simulate_backtest
            logs = fetch_historical_data()
            if not logs.empty:
                st.session_state.backtest = simulate_backtest(logs, bankroll=100)
        except:
            pass

# ─── SIDEBAR ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 8px;">
        <span style="font-size: 2.5rem;">⚡</span>
        <h2 style="margin: 0; background: linear-gradient(90deg, #388bfd, #a371f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Kalshi Quant Scanner</h2>
        <p style="color: #8b949e; font-size: 0.9rem;">Vol · Z-Score · Kelly · Weather Arb</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 REFRESH ALL MARKETS", type="primary", use_container_width=True):
        with st.spinner("📡 Fetching fresh market data..."):
            scanner = st.session_state.scanner
            scanner.fetch_markets()
            st.session_state.scan_time = datetime.now(timezone.utc)
            # Re-run non-quant analyses (quant comes from Azure Table)
            st.session_state.weather_arb = scanner.scan_weather_arb()
            st.session_state.fred = scanner.scan_fred()
            st.session_state.smart = scanner.scan_smart_money()
            st.session_state.weather_raw = scanner.scan_weather_raw()
            st.session_state.arb = scanner.scan_arbitrage()
            st.session_state['yield'] = scanner.scan_yield_farms()
            st.rerun()

    if st.session_state.scan_time:
        st.caption(f"Last scan: {st.session_state.scan_time.strftime('%H:%M:%S UTC')}")

    st.markdown("---")
    show_explanations = st.toggle("📖 Show Measure Explanations", value=True)
    st.markdown("---")

    # ── Backtest Panel ──
    with st.expander("📊 Backtest Performance", expanded=False):
        if show_explanations:
            st.markdown("""
            <div class="explain-box" style="font-size: 0.78rem;">
            <strong>What is this?</strong><br>
            The backtester replays your model's <em>historical predictions</em> from Azure logs
            and simulates what would have happened if you traded them with Kelly sizing.<br><br>
            <b>Win Rate</b> — % of trades that were correct<br>
            <b>Sharpe</b> — Risk-adjusted return (>1 = good, >2 = great)<br>
            <b>Max DD</b> — Worst peak-to-trough drawdown<br>
            <b>Profit Factor</b> — Gross wins / gross losses (>1 = profitable)<br>
            <b>Direction Accuracy</b> — How often the ML model guesses the right direction (up/down)
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔄 Reload Backtest", use_container_width=True):
            with st.spinner("Fetching Azure logs..."):
                try:
                    from src.backtester import fetch_historical_data, simulate_backtest
                    logs = fetch_historical_data()
                    if not logs.empty:
                        st.session_state.backtest = simulate_backtest(logs, bankroll=100)
                    else:
                        st.warning("No historical data in Azure.")
                except Exception as e:
                    st.error(f"Backtest error: {e}")

        if 'backtest' in st.session_state and st.session_state.backtest:
            bt = st.session_state.backtest
            m = bt['metrics']
            acc = bt.get('accuracy', {})

            cols = st.columns(2)
            cols[0].metric("Win Rate", f"{m['win_rate']}%")
            cols[1].metric("Total P&L", f"${m['total_pnl']}")
            cols = st.columns(2)
            cols[0].metric("Sharpe", f"{m['sharpe']}")
            cols[1].metric("Max DD", f"{m['max_drawdown']}%")
            cols = st.columns(2)
            cols[0].metric("Trades", m['total_trades'])
            cols[1].metric("Profit Factor", f"{m['profit_factor']}")

            if acc:
                st.markdown("---")
                st.caption("**Model Accuracy**")
                st.metric("Direction Accuracy", f"{acc.get('direction_accuracy', 0)}%")
                st.caption(f"{acc.get('direction_correct', 0)}/{acc.get('direction_total', 0)} predictions correct")
                st.caption(f"Edge filter: >{acc.get('min_edge_used', 10)}% ({acc.get('trades_filtered', 0)} skipped)")

            if bt['equity_curve']:
                eq_df = pd.DataFrame(bt['equity_curve'], columns=['Time', 'Equity'])
                eq_df['Time'] = pd.to_datetime(eq_df['Time'])
                st.line_chart(eq_df.set_index('Time')['Equity'])

# ─── HERO HEADER ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
<h1>⚡ Kalshi Quant Scanner</h1>
<p>Black-Scholes Probability · Fractional Kelly · Weather Arb · FRED Economics</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN CONTENT ────────────────────────────────────────────────────
scanner = st.session_state.scanner

if True:  # Always show main content — no blocking gate
    # ── Metrics Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    quant_opps, quant_last_updated = fetch_opportunities()
    market_count = len(scanner.markets) if scanner.markets else 0
    c1.metric("📡 Markets", f"{market_count:,}")
    c2.metric("⚡ Quant", len(quant_opps))
    c3.metric("⛈️ Weather Arb", len(st.session_state.weather_arb or []))
    c4.metric("💰 Arb", len(st.session_state.arb or []))
    c5.metric("🌾 Yield", len(st.session_state.get('yield') or []))

    if scanner.markets:
        cats = {}
        for m in scanner.markets:
            cats[m['category']] = cats.get(m['category'], 0) + 1
        pills_html = " ".join([f'<span class="stat-pill">{k}: {v}</span>' for k, v in sorted(cats.items(), key=lambda x: -x[1])])
        st.markdown(f'<div style="margin-bottom: 16px;">{pills_html}</div>', unsafe_allow_html=True)
    else:
        st.caption("💡 Click **REFRESH ALL MARKETS** in the sidebar to load Weather, FRED, and other live tabs.")

    # ─── TABS ────────────────────────────────────────────────────
    tab_quant, tab_weather, tab_smart, tab_free = st.tabs([
        "⚡ Hourly Scalps (Quant)",
        "⛈️ Weather Arb",
        "🏛️ Smart Money + FRED",
        "💸 Free Money"
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1: QUANT — Vol + Black-Scholes + Kelly
    # ═══════════════════════════════════════════════════════════════
    with tab_quant:
        qcol1, qcol2 = st.columns([6, 1])
        qcol1.markdown("### ⚡ Quantitative Financial Signals")
        if qcol2.button("🔄", key="refresh_quant", help="Re-read latest from Azure"):
            st.cache_data.clear()
            st.rerun()

        # Status bar: last updated timestamp
        if quant_last_updated:
            st.caption(f"📡 Last updated: {quant_last_updated} · Auto-refreshes every 15 mins via GitHub Actions")
        else:
            st.warning("⏳ Waiting for Background Scanner... No data in Azure Table yet.")

        st.caption("Black-Scholes probability + ML drift + fractional Kelly sizing. Max bet: $20.")

        if show_explanations:
            st.markdown("""
            <div class="explain-box">
            <strong>How This Works:</strong><br>
            <b>① Volatility (σ)</b> — Hourly std dev of log returns over 24 periods. Measures how much the asset moves per hour.<br>
            <b>② Drift (μ)</b> — Your LightGBM model predicts the next-hour price. Drift = (Pred - Current) / Current.<br>
            <b>③ Z-Score</b> — Standardized distance: (ln(Price/Strike) + Drift) / σ√t. Positive = likely above strike.<br>
            <b>④ My Prob</b> — CDF(Z) × 100. Our calculated probability the asset finishes above the strike.<br>
            <b>⑤ Edge</b> — My Prob − Kalshi Price. Positive edge = market is underpricing the outcome.<br>
            <b>⑥ Kelly Bet</b> — Quarter-Kelly sizing: only risk what the math says. Capped at $20.
            </div>
            """, unsafe_allow_html=True)

        if quant_opps:
            for sig in quant_opps:
                edge_val = float(sig.get('Edge', 0))
                edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
                edge_sign = "+" if edge_val > 0 else ""
                kelly_val = float(sig.get('KellySuggestion', 0))
                curr_price = float(sig.get('CurrentPrice', 0))
                model_pred = float(sig.get('ModelPred', 0))
                strike_val = sig.get('Strike', '0')
                vol_val = float(sig.get('Volatility', 0))
                confidence = float(sig.get('Confidence', 0))
                hours_left = float(sig.get('HoursLeft', 0))

                st.markdown(f"""
                <div class="quant-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #c9d1d9; font-size: 1.05rem;">{sig.get('Market', '')}</strong>
                            <span class="stat-pill">{sig.get('Asset', '')}</span>
                            <span class="stat-pill">{sig.get('Action', '')}</span>
                        </div>
                        <div style="text-align: right;">
                            <span class="{edge_class}" style="font-size: 1.3rem;">{edge_sign}{edge_val:.1f}%</span>
                            <br><span style="color: #8b949e; font-size: 0.8rem;">Kelly: ${kelly_val:.2f}</span>
                        </div>
                    </div>
                    <div style="margin-top: 8px; display: flex; gap: 16px; font-size: 0.85rem; color: #8b949e;">
                        <span>📊 Price: ${curr_price:,.2f}</span>
                        <span>🎯 Pred: ${model_pred:,.2f}</span>
                        <span>⚡ Strike: {strike_val}</span>
                        <span>σ: {vol_val:.5f}</span>
                        <span>P(>{strike_val}): {confidence:.1f}%</span>
                        <span>⏱ {hours_left:.1f}h</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            if quant_last_updated:
                st.info("No high-conviction quant signals at this time. The background scanner found no opportunities with >5% edge.")
            else:
                st.info("Background scanner hasn't run yet. Push to GitHub and trigger the workflow, or run `python scripts/background_scanner.py` locally.")

    # ═══════════════════════════════════════════════════════════════
    # TAB 2: WEATHER ARB
    # ═══════════════════════════════════════════════════════════════
    with tab_weather:
        wcol1, wcol2 = st.columns([6, 1])
        wcol1.markdown("### ⛈️ Weather Arbitrage")
        if wcol2.button("🔄", key="refresh_weather", help="Refresh weather arb only"):
            with st.spinner("Re-scanning weather markets..."):
                st.session_state.weather_arb = scanner.scan_weather_arb()
                st.session_state.weather_raw = scanner.scan_weather_raw()
                st.rerun()

        st.caption("Open-Meteo forecast vs Kalshi market prices. Pure math — no ML guessing.")

        if show_explanations:
            st.markdown("""
            <div class="explain-box">
            <strong>How This Works:</strong><br>
            <b>① Forecast</b> — Open-Meteo API provides free, accurate weather forecasts (temp, rain, snow) for 16 cities.<br>
            <b>② Strike</b> — Kalshi weather markets ask "Will temp be above X°?" or "Will it rain?"<br>
            <b>③ Edge</b> — Our probability (from forecast) minus Kalshi's price. >15% = actionable trade.<br>
            <b>④ Why It Works</b> — Retail Kalshi bettors don't check professional weather models. Pure information arb.
            </div>
            """, unsafe_allow_html=True)

        weather_arb = st.session_state.weather_arb or []
        if weather_arb:
            for w in weather_arb:
                edge_sign = "+" if w['edge'] > 0 else ""
                st.markdown(f"""
                <div class="weather-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #c9d1d9;">{w['title']}</strong>
                            <span class="stat-pill">🌡 {w['type']}</span>
                            <span class="stat-pill">📍 {w['city']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span class="edge-positive" style="font-size: 1.3rem;">{edge_sign}{w['edge']:.1f}%</span>
                            <br><span style="color: #8b949e;">{w['action']}</span>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.85rem; color: #8b949e;">
                        🌤 Forecast: {w['forecast']} · Strike: {w['strike']} ·
                        Our Prob: {w['my_prob']}% · Market: {w['mkt_prob']}¢ ·
                        Vol: {w.get('volume', 0):,}
                    </div>
                    <a href="{w.get('kalshi_url', '#')}" target="_blank" class="kalshi-link">🔗 Trade on Kalshi</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No weather mispricings found (>15% edge required). Markets may be efficiently priced right now.")

        st.markdown("---")
        st.markdown("#### 🌍 All Weather Markets")
        weather_raw = st.session_state.weather_raw or []
        if weather_raw:
            wdf = pd.DataFrame(weather_raw)[['title', 'price', 'volume', 'ticker']]
            wdf.columns = ['Market', 'Price (¢)', 'Volume', 'Ticker']
            search = st.text_input("🔍 Filter weather markets", key="weather_search")
            if search:
                wdf = wdf[wdf['Market'].str.contains(search, case=False)]
            st.dataframe(wdf, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 3: SMART MONEY + FRED
    # ═══════════════════════════════════════════════════════════════
    with tab_smart:
        scol1, scol2 = st.columns([6, 1])
        scol1.markdown("### 🏛️ Smart Money + FRED Economics")
        if scol2.button("🔄", key="refresh_fred", help="Refresh FRED data only"):
            with st.spinner("Refreshing FRED + Smart Money..."):
                st.session_state.scanner.fred_dashboard = None  # Force re-fetch
                st.session_state.fred = scanner.scan_fred()
                st.session_state.smart = scanner.scan_smart_money()
                st.rerun()

        fred_data = st.session_state.fred or {}
        dashboard = fred_data.get('dashboard', {})

        if dashboard:
            st.markdown("#### 📈 FRED Macro Dashboard")
            if show_explanations:
                st.markdown("""
                <div class="explain-box">
                <strong>Real-Time FRED Data:</strong> Federal Reserve Economic Data provides official macro indicators.
                These are compared against Kalshi economic markets to find where the market diverges from official data.
                </div>
                """, unsafe_allow_html=True)

            mcols = st.columns(5)
            for i, key in enumerate(['fed_rate', 'unemployment', 'inflation_exp', 'treasury_10y', 'vix']):
                if key in dashboard and dashboard[key]['value'] is not None:
                    d = dashboard[key]
                    delta = f"{d['change']:+.2f}" if d['change'] is not None else None
                    mcols[i % 5].metric(
                        d['name'].split('(')[0].strip()[:20],
                        f"{d['value']:.2f}{d['unit'] if d['unit'] != 'Index' else ''}",
                        delta=delta
                    )

            mcols2 = st.columns(5)
            for i, key in enumerate(['treasury_2y', 'consumer_sent', 'gdp_growth', 'debt_gdp']):
                if key in dashboard and dashboard[key]['value'] is not None:
                    d = dashboard[key]
                    delta = f"{d['change']:+.2f}" if d['change'] is not None else None
                    mcols2[i % 5].metric(
                        d['name'].split('(')[0].strip()[:20],
                        f"{d['value']:.1f}{d['unit'] if d['unit'] not in ['Index', 'Billions $', 'Thousands'] else ''}",
                        delta=delta
                    )

            yc = fred_data.get('yield_curve')
            if yc:
                st.markdown("---")
                yc_cols = st.columns(4)
                yc_cols[0].metric("10Y Yield", f"{yc['latest_10y']:.2f}%")
                yc_cols[1].metric("2Y Yield", f"{yc['latest_2y']:.2f}%")
                yc_cols[2].metric("Spread", f"{yc['spread']:.2f}%")
                yc_cols[3].metric("Inverted?", "⚠️ YES" if yc['inverted'] else "✅ NO")
                if yc['history']:
                    yc_df = pd.DataFrame(yc['history'])
                    yc_df['date'] = pd.to_datetime(yc_df['date'])
                    st.line_chart(yc_df.set_index('date')['spread'], use_container_width=True)
            st.markdown("---")

        fred_analysis = fred_data.get('analysis', [])
        if fred_analysis:
            st.markdown("#### 🔬 FRED vs Kalshi Analysis")
            for a in fred_analysis[:20]:
                st.markdown(f"""
                <div class="fred-card">
                    <strong style="color: #c9d1d9;">{a['market']}</strong>
                    <span class="stat-pill">{a['category']}</span>
                    <div style="margin-top: 6px; font-size: 0.85rem; color: #8b949e;">
                        📊 {a['indicator_name']}: <strong>{a['current_indicator']}</strong> ·
                        Kalshi: {a['kalshi_price']}¢ · Vol: {a.get('volume', 0):,}<br>
                        💡 {a['insight']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

        st.markdown("#### 🏛️ Economic & Political Markets")
        smart = st.session_state.smart or []
        if smart:
            sm_df = pd.DataFrame(smart)[['title', 'category', 'price', 'volume', 'ticker']]
            sm_df.columns = ['Market', 'Category', 'Price (¢)', 'Volume', 'Ticker']
            cat_filter = st.multiselect("Filter by category", sm_df['Category'].unique(), default=sm_df['Category'].unique())
            search = st.text_input("🔍 Search markets", key="smart_search")
            filtered = sm_df[sm_df['Category'].isin(cat_filter)]
            if search:
                filtered = filtered[filtered['Market'].str.contains(search, case=False)]
            st.dataframe(filtered.sort_values('Volume', ascending=False), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: FREE MONEY
    # ═══════════════════════════════════════════════════════════════
    with tab_free:
        fcol1, fcol2 = st.columns([6, 1])
        fcol1.markdown("### 💸 Free Money")
        if fcol2.button("🔄", key="refresh_free", help="Refresh arb + yield"):
            with st.spinner("Refreshing arb + yield..."):
                st.session_state.arb = scanner.scan_arbitrage()
                st.session_state['yield'] = scanner.scan_yield_farms()
                st.rerun()

        # ── ARBITRAGE ──
        st.markdown("#### 🎯 Arbitrage Detection")
        st.caption("Markets where `Cost(YES) + Cost(NO) < $1.00` — guaranteed profit.")

        if show_explanations:
            st.markdown("""
            <div class="explain-box">
            <strong>How This Works:</strong><br>
            Binary outcomes sum to 100%. If YES + NO < 100¢, buying both guarantees profit.
            Example: YES at 45¢ + NO at 52¢ = 97¢ cost → $3 guaranteed profit per contract.
            </div>
            """, unsafe_allow_html=True)

        arb = st.session_state.arb or []
        if arb:
            for a in arb[:10]:
                st.markdown(f"""
                <div class="quant-card">
                    <strong style="color: #3fb950;">{a['title']}</strong>
                    <span class="stat-pill">{a['category']}</span>
                    <div style="margin-top: 6px; font-size: 0.85rem; color: #8b949e;">
                        YES: {a['price']}¢ + NO: {a['no_price']}¢ = {a['cost']}¢ →
                        <span class="edge-positive">FREE ${(a['profit'] / 100):.2f}</span>
                    </div>
                    <a href="{a.get('kalshi_url', '#')}" target="_blank" class="kalshi-link">🔗 Trade on Kalshi</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No arbitrage opportunities. Market is efficient (YES + NO ≥ $1.00 everywhere).")

        st.markdown("---")

        # ── YIELD FARMING ──
        st.markdown("#### 🌾 Yield Farming")
        st.caption("High-probability bets (92-98¢) expiring within 48h. Low risk, steady returns.")

        if show_explanations:
            st.markdown("""
            <div class="explain-box">
            <strong>How This Works:</strong><br>
            Markets at 92-98¢ are almost certainly going to resolve YES. Buying at 95¢ pays 100¢ = 5.3% ROI.
            Only picks markets expiring within 48h to minimize time-risk. Sports excluded.
            </div>
            """, unsafe_allow_html=True)

        yf = st.session_state.get('yield') or []
        if yf:
            for y in yf[:15]:
                st.markdown(f"""
                <div class="quant-card">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <strong style="color: #c9d1d9;">{y['title']}</strong>
                            <span class="stat-pill">{y['category']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span class="edge-positive">{y['roi']:.1f}% ROI</span>
                            <br><span style="color: #8b949e; font-size: 0.8rem;">⏱ {y['hours_left']}h left</span>
                        </div>
                    </div>
                    <div style="margin-top: 4px; font-size: 0.85rem; color: #8b949e;">
                        Buy at {y['price']}¢ → Payout 100¢ · Vol: {y.get('volume', 0):,}
                    </div>
                    <a href="{y.get('kalshi_url', '#')}" target="_blank" class="kalshi-link">🔗 Trade on Kalshi</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No yield farming opportunities found. No 92-98¢ bets expiring within 48 hours.")

