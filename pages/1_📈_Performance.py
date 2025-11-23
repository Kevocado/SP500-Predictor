import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add src to path so we can import modules
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Actually, since we are in 'pages/', we need to go up one level to find 'src' if we were running this directly.
# But Streamlit runs from the root, so 'src' should be importable if the root is in path.
# Let's assume standard Streamlit behavior where running `streamlit run streamlit_app.py` makes the root importable.

from src.data_loader import fetch_data
from src.model import load_model
from src.evaluation import evaluate_model
from src.model_daily import load_daily_model
from src.utils import determine_best_timeframe

# Azure Logger with Safety Wrapper
try:
    from src.azure_logger import fetch_all_logs
    AZURE_AVAILABLE = True
except Exception as e:
    AZURE_AVAILABLE = False
    # print(f"⚠️ Azure logging disabled: {e}")

st.set_page_config(page_title="Performance Analytics", layout="wide")

# --- SHARED STATE ---
if 'selected_asset' in st.session_state:
    selected_ticker = st.session_state.selected_asset
else:
    selected_ticker = "SPX" # Default

st.title(f"📈 Performance Analysis: {selected_ticker}")

# Determine timeframe
recommended_tf = determine_best_timeframe(selected_ticker)
timeframe_map = {"Hourly": "Hourly", "Daily": "Daily"}
internal_timeframe = timeframe_map.get(recommended_tf, "Daily")

st.caption(f"Historical performance metrics for {selected_ticker} ({internal_timeframe})")

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
