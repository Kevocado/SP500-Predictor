"""
Kalshi Quant Scanner — v3 (Multi-Engine System)
Pure UI frontend that reads computed ops from Azure Table Storage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

# ─── AZURE TABLE: FETCH PRE-COMPUTED OPPORTUNITIES ────────────
@st.cache_data(ttl=30)
def fetch_opportunities():
    """Reads from Azure Table Storage and separates Live vs Paper."""
    try:
        from azure.data.tables import TableClient
        conn_str = os.getenv("AZURE_CONNECTION_STRING")
        if not conn_str:
            return [], [], None

        client = TableClient.from_connection_string(conn_str, "CurrentOpportunities")
        
        entities = list(client.query_entities(""))
        
        # Sort by edge
        entities.sort(key=lambda x: float(x.get('Edge', 0)), reverse=True)
        
        live_opps = [e for e in entities if e.get('PartitionKey') == 'Live']
        paper_opps = [e for e in entities if e.get('PartitionKey') == 'Paper']

        # Get last-updated timestamp
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

        return live_opps, paper_opps, last_update
    except Exception as e:
        print(f"fetch_opportunities error: {e}")
        return [], [], None

st.set_page_config(
    page_title="Kalshi Quant Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── FULL DARK THEME CSS ─────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
<h1>⚡ Kalshi Quant Scanner</h1>
<p>Multi-Engine Prediction Market System (Weather, Macro, Quant Lab)</p>
</div>
""", unsafe_allow_html=True)

live_opps, paper_opps, last_updated = fetch_opportunities()

col1, col2, col3 = st.columns(3)
col1.metric("🔴 Real Edge (Live) Opps", len(live_opps))
col2.metric("🧪 Quant Lab (Paper) Opps", len(paper_opps))
col3.markdown(f"**Last Sync:** {last_updated or 'N/A'}")

if st.button("🔄 Force Refresh Storage Cache", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")

tab_live, tab_paper = st.tabs([
    "🔴 Real Edge (Live)",
    "🧪 Quant Lab (Paper Trading)"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: REAL EDGE (LIVE)
# ═══════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### High-Conviction Weather and Macro Trades")
    st.caption("These engines use hard deterministic data against Kalshi markets. Suitable for live capital deployment.")
    
    if live_opps:
        for sig in live_opps:
            engine = sig.get('Engine', 'Unknown')
            edge_val = float(sig.get('Edge', 0))
            edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
            edge_sign = "+" if edge_val > 0 else ""
            
            st.markdown(f"""
            <div class="quant-card">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong style="color: #c9d1d9; font-size: 1.05rem;">{sig.get('Market', sig.get('Asset', 'Unknown Market'))}</strong>
                        <span class="stat-pill">{engine} Engine</span>
                        <span class="stat-pill">{sig.get('Action', 'TRADE')}</span>
                    </div>
                    <div style="text-align: right;">
                        <span class="{edge_class}" style="font-size: 1.3rem;">{edge_sign}{edge_val:.1f}%</span>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.85rem; color: #8b949e;">
                    <p><b>Analysis:</b> {sig.get('Reasoning', sig.get('Analysis', 'No detailed reasoning provided.'))}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No Live edge opportunities found currently. Engines are searching.")

# ═══════════════════════════════════════════════════════════════
# TAB 2: QUANT LAB (PAPER TRADING)
# ═══════════════════════════════════════════════════════════════
with tab_paper:
    st.markdown("### SPX/BTC Algorithmic Predictions (Paper Only)")
    st.caption("Purely math-driven algorithmic system. Black-Scholes + LightGBM volatility prediction.")
    
    if paper_opps:
        for sig in paper_opps:
            edge_val = float(sig.get('Edge', 0))
            edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
            edge_sign = "+" if edge_val > 0 else ""
            kelly_val = float(sig.get('KellySuggestion', 0))
            
            st.markdown(f"""
            <div class="quant-card" style="border-left: 4px solid #a371f7;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong style="color: #c9d1d9; font-size: 1.05rem;">{sig.get('Market', 'Unknown Market')}</strong>
                        <span class="stat-pill">{sig.get('Asset', 'Crypto/Equities')}</span>
                        <span class="stat-pill" style="border-color: #a371f7; color: #a371f7!important;">PAPER TRADE ONLY</span>
                    </div>
                    <div style="text-align: right;">
                        <span class="{edge_class}" style="font-size: 1.3rem;">{edge_sign}{edge_val:.1f}%</span>
                        <br><span style="color: #8b949e; font-size: 0.8rem;">Kelly: ${kelly_val:.2f}</span>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.85rem; color: #8b949e; display: flex; gap: 16px;">
                    <span>📊 Price: ${float(sig.get('CurrentPrice', 0)):,.2f}</span>
                    <span>🎯 Pred: ${float(sig.get('ModelPred', 0)):,.2f}</span>
                    <span>⚡ Strike: {sig.get('Strike', '0')}</span>
                    <span>σ: {float(sig.get('Volatility', 0)):.5f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No Paper Trading signals found currently. Mathematical models found no edge >5%.")
