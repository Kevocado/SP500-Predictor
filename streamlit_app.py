"""
Kalshi Pro Scanner
4-tab interface: AI Predictor, Sniper, Weather, Politics & Econ
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from src.market_scanner import HybridScanner
from src.sentiment import render_sentiment_panel

# ─── PAGE CONFIG ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kalshi Pro Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── PREMIUM DARK THEME CSS ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(48, 54, 61, 0.6);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.5);
        padding: 12px 16px;
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13, 17, 23, 0.6);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 139, 253, 0.15);
        border-bottom: 2px solid #388bfd;
    }

    /* Hero */
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
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .hero-header p {
        color: #8b949e;
        margin: 4px 0 0;
        font-size: 0.95rem;
    }

    /* Alert cards */
    .arb-alert {
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.08) 0%, rgba(0, 200, 120, 0.05) 100%);
        border: 1px solid rgba(0, 255, 157, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .arb-profit { color: #00ff9d; font-size: 1.8rem; font-weight: 700; }

    .yield-card {
        background: linear-gradient(135deg, rgba(255, 170, 0, 0.06) 0%, rgba(255, 200, 50, 0.03) 100%);
        border: 1px solid rgba(255, 170, 0, 0.25);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .yield-roi { color: #ffaa00; font-size: 1.4rem; font-weight: 700; }

    .ml-signal {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.08) 0%, rgba(100, 160, 255, 0.04) 100%);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    .stat-pill {
        display: inline-block;
        background: rgba(56, 139, 253, 0.12);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #388bfd;
        margin-right: 6px;
    }
    .stat-pill-green { background: rgba(0,255,157,0.1); border-color: rgba(0,255,157,0.3); color: #00ff9d; }
    .stat-pill-amber { background: rgba(255,170,0,0.1); border-color: rgba(255,170,0,0.3); color: #ffaa00; }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #8b949e;
    }
    .empty-state h2 { color: #c9d1d9; margin-bottom: 8px; }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #388bfd, #a371f7);
    }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────
if 'scanner' not in st.session_state:
    st.session_state.scanner = HybridScanner()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'scan_time' not in st.session_state:
    st.session_state.scan_time = None

# ─── SIDEBAR ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 8px;">
        <span style="font-size: 2.5rem;">⚡</span>
        <h2 style="margin: 4px 0 0; background: linear-gradient(90deg, #388bfd, #a371f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Kalshi Pro</h2>
        <p style="color: #8b949e; font-size: 0.85rem; margin-top: 2px;">Deep Scan (10k Markets) Active</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔄 RUN DEEP SCAN", type="primary", use_container_width=True):
        with st.spinner("Fetching 10,000 markets to bypass sports..."):
            st.session_state.data = st.session_state.scanner.run_scan()
            st.session_state.scan_time = datetime.now(timezone.utc)
            st.rerun()

    if st.session_state.scan_time:
        st.caption(f"🕐 Last scan: {st.session_state.scan_time.strftime('%H:%M:%S UTC')}")
        total = len(st.session_state.scanner.markets)
        non_sports = len([m for m in st.session_state.scanner.markets if m['category'] != 'Sports'])
        st.caption(f"📡 {total:,} markets ({non_sports:,} non-sports)")

    st.markdown("---")
    st.info("Scanner filters out Sports by default to find 'Money Markets' (Econ, Politics, Weather).")

    show_sentiment = st.toggle("📊 Show Sentiment", value=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>⚡ Kalshi Pro Scanner</h1>
    <p>Deep Scan • AI Predictor • Smart Money • Weather • Yield Farming</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN CONTENT ────────────────────────────────────────────────────
if st.session_state.data:
    data = st.session_state.data
    scanner = st.session_state.scanner

    # ── Metrics Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📡 Markets", f"{len(scanner.markets):,}")
    c2.metric("🤖 AI Signals", len(data['ml_alpha']))
    c3.metric("⛈️ Weather", len(data['weather']))
    c4.metric("🎯 Arbitrage", len(data['arbitrage']),
              delta="FREE MONEY" if data['arbitrage'] else None,
              delta_color="normal" if data['arbitrage'] else "off")
    c5.metric("💰 Yield Farms", len(data['yield_farming']))

    st.markdown("")

    # ── Category breakdown ──
    cats = {}
    for m in scanner.markets:
        cats[m['category']] = cats.get(m['category'], 0) + 1
    cat_pills = " ".join([f'<span class="stat-pill">{cat}: {count}</span>' for cat, count in sorted(cats.items(), key=lambda x: -x[1])])
    st.markdown(f'<div style="margin-bottom: 16px;">{cat_pills}</div>', unsafe_allow_html=True)

    # ── Tabs ──
    tab_ml, tab_sniper, tab_weather, tab_macro = st.tabs([
        "📈 AI Predictor",
        "💸 Sniper (Easy Money)",
        "⛈️ Weather",
        "🏛️ Politics & Econ"
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1: AI PREDICTOR — ML Signals
    # ═══════════════════════════════════════════════════════════════
    with tab_ml:
        st.markdown("### 🤖 Machine Learning Signals")
        st.caption("Predictions based on your trained LightGBM models for SPX, Nasdaq, BTC, ETH.")

        if data['ml_alpha']:
            for sig in data['ml_alpha']:
                st.markdown(f"""
                <div class="ml-signal">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.05rem; color: #c9d1d9;">{sig['Market']}</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">
                                Asset: {sig['Asset']} • Model Prediction: {sig['Model_Pred']:.2f} • Strike: {sig['Strike']:.0f}
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #00ff9d; font-size: 1.2rem; font-weight: 700;">{sig['Action']}</span><br>
                            <span style="color: #8b949e; font-size: 0.8rem;">
                                Kalshi: {sig['Kalshi_Price']}¢ • Edge: {sig['Edge']} • Conf: {sig['Confidence']}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong ML signals found matching current Kalshi markets. This can happen when models aren't trained yet or no markets match the predictions.")
            st.caption("**Models checked:** SPX, BTC, ETH, Nasdaq — run `python -c 'from src.model import train_model; ...'` to train.")

    # ═══════════════════════════════════════════════════════════════
    # TAB 2: SNIPER — Arbitrage + Yield Farming
    # ═══════════════════════════════════════════════════════════════
    with tab_sniper:
        # ── ARBITRAGE ──
        st.markdown("### 🎯 Arbitrage Detection")
        st.caption("Markets where `Cost(YES) + Cost(NO) < $1.00` — guaranteed profit.")

        if data['arbitrage']:
            st.error(f"🚨 **{len(data['arbitrage'])} ARBITRAGE OPPORTUNITIES DETECTED**")
            for arb in data['arbitrage']:
                st.markdown(f"""
                <div class="arb-alert">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.05rem; color: #c9d1d9;">{arb['title']}</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">
                                Buy YES ({arb['price']}¢) + NO ({arb['no_price']}¢) = {arb['cost']}¢
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <div class="arb-profit">+{arb['profit']}¢</div>
                            <span style="color: #8b949e; font-size: 0.8rem;">per share guaranteed</span>
                        </div>
                    </div>
                    <div style="margin-top: 8px;">
                        <span class="stat-pill-green">Vol: {arb['volume']:,}</span>
                        <span class="stat-pill">{arb['category']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ No arbitrage mispricing detected. Market is efficient right now.")

        st.markdown("---")

        # ── YIELD FARMING ──
        st.markdown("### 💰 Yield Farming (Non-Sports)")
        st.caption("High probability (>92%) trades expiring in <48h. Economics, Politics, Weather only.")

        if data['yield_farming']:
            for farm in data['yield_farming']:
                st.markdown(f"""
                <div class="yield-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #c9d1d9;">{farm['title']}</strong><br>
                            <span style="color: #8b949e; font-size: 0.85rem;">
                                {farm['category']} • Vol: {farm['volume']:,}
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <div class="yield-roi">+{farm['roi']:.1f}%</div>
                            <span style="color: #8b949e; font-size: 0.8rem;">
                                Cost: {farm['price']}¢ • {farm['hours_left']}h left
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No yield farms found. Markets may be too far from expiry or no high-probability non-sports bets available.")

    # ═══════════════════════════════════════════════════════════════
    # TAB 3: WEATHER — Climate & Weather Markets
    # ═══════════════════════════════════════════════════════════════
    with tab_weather:
        st.markdown("### ⛈️ Climate & Weather")
        st.caption("Hurricanes, Temperature, Rain, Snow — bet on the weather.")

        if data['weather']:
            w_search = st.text_input("🔍 Filter Weather (e.g. 'NYC', 'Rain', 'Temperature')", "", key="weather_search")

            filtered_w = [w for w in data['weather'] if w_search.lower() in w['title'].lower()] if w_search else data['weather']

            st.caption(f"Showing {len(filtered_w)} of {len(data['weather'])} weather markets")

            for w in filtered_w:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([4, 1, 1])
                    with c1:
                        st.markdown(f"**{w['title']}**")
                        st.caption(f"Expires: {w['expiration'] or 'N/A'}")
                    with c2:
                        pct = min(w['price'] / 100, 1.0)
                        st.progress(pct, text=f"Prob: {w['price']}%")
                    with c3:
                        st.metric("Volume", f"{w['volume']:,}")
        else:
            st.info("No active weather markets found. These may appear seasonally (hurricane season, extreme weather events).")

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: MACRO — Economics & Politics
    # ═══════════════════════════════════════════════════════════════
    with tab_macro:
        st.markdown("### 🏛️ Smart Money (Economics & Politics)")
        st.caption("Fed Rates, CPI, GDP, Elections — where the smart money plays.")

        if data['smart_money']:
            # Sub-filter
            econ_only = [m for m in data['smart_money'] if m['category'] == 'Economics']
            pol_only = [m for m in data['smart_money'] if m['category'] == 'Politics']

            e1, e2 = st.columns(2)
            e1.metric("📊 Economics", len(econ_only))
            e2.metric("🏛️ Politics", len(pol_only))

            for m in data['smart_money']:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([4, 1, 1])
                    with c1:
                        st.markdown(f"**{m['title']}**")
                        st.caption(f"{m['category']} • `{m['ticker']}`")
                    with c2:
                        pct = min(m['price'] / 100, 1.0)
                        st.progress(pct, text=f"Prob: {m['price']}%")
                    with c3:
                        st.metric("Volume", f"{m['volume']:,}")
        else:
            st.info("No active Economics or Politics markets found.")

    # ── Sentiment Panel ──
    if show_sentiment:
        st.markdown("---")
        with st.expander("📊 Market Sentiment", expanded=False):
            render_sentiment_panel(ticker="SPX")

# ─── EMPTY STATE ─────────────────────────────────────────────────────
else:
    st.markdown("")
    st.markdown("""
    <div class="empty-state">
        <h2>👋 Ready to Deep Scan</h2>
        <p style="font-size: 1.1rem; margin-bottom: 20px;">
            Click <strong>🔄 RUN DEEP SCAN</strong> in the sidebar to fetch 10k markets and filter out the noise.
        </p>
        <div style="display: flex; justify-content: center; gap: 32px; margin-top: 24px;">
            <div style="text-align: center;">
                <span style="font-size: 2rem;">📈</span>
                <p style="margin-top: 4px;"><strong>AI Predictor</strong></p>
                <p style="font-size: 0.85rem;">SPX • BTC • ETH • Nasdaq</p>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 2rem;">💸</span>
                <p style="margin-top: 4px;"><strong>Sniper</strong></p>
                <p style="font-size: 0.85rem;">Arb • Yield Farming</p>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 2rem;">⛈️</span>
                <p style="margin-top: 4px;"><strong>Weather</strong></p>
                <p style="font-size: 0.85rem;">Climate • Temperature</p>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 2rem;">🏛️</span>
                <p style="margin-top: 4px;"><strong>Politics & Econ</strong></p>
                <p style="font-size: 0.85rem;">Fed • CPI • Elections</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_sentiment:
        st.markdown("---")
        with st.expander("📊 Market Sentiment", expanded=True):
            render_sentiment_panel(ticker="SPX")

# ─── FOOTER ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice. Trading involves risk.")
