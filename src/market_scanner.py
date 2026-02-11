"""
Kalshi Market Scanner Module
Scans Kalshi prediction markets across multiple assets,
calculates fair value using existing ML models, detects edges,
and renders signal cards in Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import logging

from src.kalshi_feed import get_real_kalshi_markets
from src.data_loader import fetch_data
from src.feature_engineering import create_features, prepare_training_data
from src.model import load_model, predict_next_hour, calculate_probability, get_recent_rmse, train_model, FeatureMismatchError
from src.model_daily import load_daily_model, predict_daily_close, prepare_daily_data
from src.utils import categorize_markets

logger = logging.getLogger(__name__)


class KalshiScanner:
    """Wraps existing Kalshi feed for multi-asset scanning."""

    TICKERS = ["SPX", "Nasdaq", "BTC", "ETH"]

    def scan_all_markets(self) -> Dict[str, List[Dict]]:
        """Fetch all Kalshi markets for every tracked asset."""
        results = {}
        for ticker in self.TICKERS:
            try:
                markets, method, debug = get_real_kalshi_markets(ticker)
                results[ticker] = {
                    "markets": markets,
                    "method": method,
                    "debug": debug,
                }
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                results[ticker] = {"markets": [], "method": "Error", "debug": {}}
        return results


class SignalGenerator:
    """Generate trading signals using existing model infrastructure."""

    def __init__(self, min_edge: float = 8.0, max_kelly: float = 0.06):
        self.min_edge = min_edge
        self.max_kelly = max_kelly

    def _load_models_and_data(self, ticker: str):
        """Load models and fetch fresh data for a ticker."""
        # Hourly
        df_hourly = fetch_data(ticker=ticker, period="5d", interval="1m")
        model_hourly, needs_retrain = load_model(ticker=ticker)

        if (needs_retrain or model_hourly is None) and not df_hourly.empty:
            try:
                df_train = fetch_data(ticker=ticker, period="7d", interval="1m")
                df_train = prepare_training_data(df_train)
                model_hourly = train_model(df_train, ticker=ticker)
            except Exception:
                model_hourly = None

        # Daily
        df_daily = fetch_data(ticker=ticker, period="60d", interval="1h")
        model_daily = load_daily_model(ticker=ticker)

        return df_hourly, model_hourly, df_daily, model_daily

    def generate_signals(self, scanner_results: Dict, bankroll: float = 1000) -> List[Dict]:
        """
        Generate signals from scanner results.
        Returns a sorted list of signal dicts.
        """
        signals = []

        for ticker, data in scanner_results.items():
            markets = data.get("markets", [])
            if not markets:
                continue

            try:
                df_hourly, model_hourly, df_daily, model_daily = self._load_models_and_data(ticker)
                buckets = categorize_markets(markets, ticker)

                # --- Process Hourly ---
                if not df_hourly.empty and model_hourly and buckets['hourly']:
                    try:
                        df_feat = create_features(df_hourly)
                        pred = predict_next_hour(model_hourly, df_feat, ticker=ticker)
                        rmse = get_recent_rmse(model_hourly, df_hourly, ticker=ticker)
                        curr_price = df_hourly['Close'].iloc[-1]
                    except FeatureMismatchError:
                        try:
                            df_train = prepare_training_data(
                                fetch_data(ticker=ticker, period="7d", interval="1m")
                            )
                            model_hourly = train_model(df_train, ticker=ticker)
                            df_feat = create_features(df_hourly)
                            pred = predict_next_hour(model_hourly, df_feat, ticker=ticker)
                            rmse = get_recent_rmse(model_hourly, df_hourly, ticker=ticker)
                            curr_price = df_hourly['Close'].iloc[-1]
                        except Exception:
                            pred = rmse = curr_price = None

                    if pred is not None:
                        for m in buckets['hourly']:
                            sig = self._evaluate_strike(m, ticker, pred, rmse, curr_price, "Hourly", bankroll)
                            if sig:
                                signals.append(sig)

                # --- Process Daily ---
                if not df_daily.empty and model_daily and buckets['daily']:
                    try:
                        df_feat_d, _ = prepare_daily_data(df_daily)
                        pred_d = predict_daily_close(model_daily, df_feat_d.iloc[[-1]])
                        rmse_d = df_daily['Close'].iloc[-1] * 0.01
                        curr_price_d = df_daily['Close'].iloc[-1]
                    except Exception:
                        pred_d = rmse_d = curr_price_d = None

                    if pred_d is not None:
                        for m in buckets['daily']:
                            sig = self._evaluate_strike(m, ticker, pred_d, rmse_d, curr_price_d, "Daily", bankroll)
                            if sig:
                                signals.append(sig)

            except Exception as e:
                logger.error(f"Error generating signals for {ticker}: {e}")

        # Sort by edge descending
        signals.sort(key=lambda x: abs(x.get('edge', 0)), reverse=True)
        return signals

    def _evaluate_strike(self, market: Dict, ticker: str, pred: float,
                         rmse: float, curr_price: float, timeframe: str,
                         bankroll: float) -> Dict | None:
        """Evaluate a single strike market and return a signal dict or None."""
        strike = market.get('strike_price')
        if not strike:
            return None

        # Moneyness filter: within 2% of current price
        if curr_price and curr_price > 0:
            if abs(strike - curr_price) / curr_price > 0.02:
                return None

        market_type = market.get('market_type', 'above')
        prob_above = calculate_probability(pred, strike, rmse)

        if market_type == 'below':
            prob_win = 100 - prob_above
            strike_label = f"< ${strike:,.0f}"
        else:
            prob_win = prob_above
            strike_label = f"> ${strike:,.0f}"

        if prob_win > 50:
            action = "BUY YES"
            conf = prob_win
        else:
            action = "BUY NO"
            conf = 100 - prob_win

        # Edge = model confidence - cost to enter (ask price)
        cost = market.get('yes_ask', 0) if "YES" in action else market.get('no_ask', 0)
        if cost <= 0:
            cost = 99
        edge = conf - cost

        if edge < self.min_edge:
            return None

        # Kelly criterion position sizing
        cost_frac = cost / 100.0
        if cost_frac < 1:
            kelly = abs(edge / 100) / (1 - cost_frac)
            kelly = min(kelly, self.max_kelly)
        else:
            kelly = 0
        position_size = bankroll * kelly

        # Expiration info
        exp_str = market.get('expiration', '')
        try:
            exp_time = pd.to_datetime(exp_str)
            time_left = exp_time.tz_localize(None) - datetime.utcnow() if exp_time.tzinfo is None else exp_time - pd.Timestamp.now(tz=exp_time.tzinfo)
            hours_left = time_left.total_seconds() / 3600
            if hours_left < 1:
                expires_in = f"{max(1, int(time_left.total_seconds() / 60))}m"
            elif hours_left < 24:
                expires_in = f"{int(hours_left)}h"
            else:
                expires_in = f"{int(hours_left / 24)}d"
        except Exception:
            expires_in = "N/A"

        return {
            'ticker': ticker,
            'strike': strike_label,
            'action': action,
            'confidence': conf,
            'market_price': cost,
            'edge': edge,
            'kelly_pct': kelly * 100,
            'position_size': position_size,
            'timeframe': timeframe,
            'expires_in': expires_in,
            'yes_bid': market.get('yes_bid', 0),
            'yes_ask': market.get('yes_ask', 0),
            'no_bid': market.get('no_bid', 0),
            'no_ask': market.get('no_ask', 0),
            'market_id': market.get('market_id', ''),
            'title': market.get('title', ''),
        }


class ScannerDashboard:
    """Renders the Kalshi Market Scanner UI inside Streamlit."""

    # Signal card CSS
    CARD_CSS = """
    <style>
        .scanner-card {
            background: rgba(26, 31, 58, 0.6);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(42, 63, 95, 0.5);
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .scanner-card:hover {
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.15);
        }
        .edge-pill {
            background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
            color: #000;
            padding: 6px 14px;
            border-radius: 16px;
            font-weight: bold;
            font-size: 1.1em;
            display: inline-block;
        }
    </style>
    """

    def __init__(self):
        self.scanner = KalshiScanner()
        self.signal_gen = SignalGenerator()

    def _init_state(self):
        if 'scanner_signals' not in st.session_state:
            st.session_state.scanner_signals = []
        if 'scanner_bankroll' not in st.session_state:
            st.session_state.scanner_bankroll = 1000
        if 'scanner_last_scan' not in st.session_state:
            st.session_state.scanner_last_scan = None
        if 'scanner_count' not in st.session_state:
            st.session_state.scanner_count = 0

    def render(self):
        """Main render entry point — call from streamlit_app.py."""
        self._init_state()
        st.markdown(self.CARD_CSS, unsafe_allow_html=True)

        # --- Header ---
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("## 🎯 Kalshi Market Scanner")
            st.caption("Scan all assets for mispriced Kalshi markets • Signal cards with edge & Kelly sizing")
        with col2:
            if st.button("🔄 Scan Markets", use_container_width=True, type="primary", key="scanner_btn"):
                self._run_scan()
                st.rerun()

        st.markdown("---")

        # --- Stats row ---
        signals = st.session_state.scanner_signals
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Bankroll", f"${st.session_state.scanner_bankroll:,.0f}")
        c2.metric("🎯 Signals", len(signals))
        avg_edge = np.mean([s['edge'] for s in signals]) if signals else 0
        c3.metric("📊 Avg Edge", f"{avg_edge:.1f}%")
        if st.session_state.scanner_last_scan:
            elapsed = (datetime.now() - st.session_state.scanner_last_scan).seconds
            c4.metric("⏱️ Last Scan", f"{elapsed}s ago")
        else:
            c4.metric("⏱️ Last Scan", "Never")

        st.markdown("---")

        # --- Sidebar controls ---
        with st.sidebar:
            st.markdown("### 🎯 Scanner Settings")
            st.session_state.scanner_bankroll = st.number_input(
                "Bankroll ($)", 100, 100000, st.session_state.scanner_bankroll, 100,
                key="scanner_bankroll_input"
            )
            min_edge = st.slider("Min Edge (%)", 0, 30, 8, key="scanner_min_edge")
            max_kelly = st.slider("Max Kelly (%)", 1, 20, 6, key="scanner_max_kelly")
            self.signal_gen.min_edge = min_edge
            self.signal_gen.max_kelly = max_kelly / 100

        # --- Signal cards ---
        if not signals:
            st.info("👆 Click **Scan Markets** to find opportunities across SPX, Nasdaq, BTC, and ETH.")
            return

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            filter_tickers = st.multiselect(
                "Filter Assets", ["SPX", "Nasdaq", "BTC", "ETH"],
                default=["SPX", "Nasdaq", "BTC", "ETH"], key="scanner_filter_assets"
            )
        with fc2:
            filter_tf = st.multiselect(
                "Filter Timeframe", ["Hourly", "Daily"],
                default=["Hourly", "Daily"], key="scanner_filter_tf"
            )

        filtered = [s for s in signals
                     if s['ticker'] in filter_tickers and s['timeframe'] in filter_tf]

        if not filtered:
            st.warning("No signals match your filters.")
            return

        st.markdown(f"### 🔥 {len(filtered)} Opportunities Found")

        for i, sig in enumerate(filtered, 1):
            self._render_card(sig, i)

    def _render_card(self, sig: Dict, index: int):
        """Render a single signal card."""
        edge_color = "#00ff88" if sig['edge'] > 15 else "#ffaa00"

        st.markdown(f"""
        <div class="scanner-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h4 style="margin:0;">#{index} {sig['ticker']} {sig['strike']}</h4>
                    <p style="color: #888; margin: 4px 0;">Kalshi • {sig['timeframe']} • Expires in {sig['expires_in']}</p>
                </div>
                <span class="edge-pill">+{sig['edge']:.1f}%</span>
            </div>
            <hr style="border-color: rgba(42,63,95,0.5); margin: 12px 0;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                <div>
                    <small style="color:#888;">Model</small><br>
                    <strong style="color:#00ff88; font-size:1.2em;">{sig['confidence']:.0f}¢</strong>
                </div>
                <div>
                    <small style="color:#888;">Market</small><br>
                    <strong style="color:#fff; font-size:1.2em;">{sig['market_price']}¢</strong>
                </div>
                <div>
                    <small style="color:#888;">Direction</small><br>
                    <strong style="color:{'#00ff88' if 'YES' in sig['action'] else '#ff5555'}; font-size:1.2em;">{sig['action']}</strong>
                </div>
                <div>
                    <small style="color:#888;">Kelly</small><br>
                    <strong style="font-size:1.2em;">${sig['position_size']:.0f} ({sig['kelly_pct']:.1f}%)</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action buttons row
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("📋 Copy Details", key=f"scan_copy_{index}", use_container_width=True):
                st.code(f"{sig['ticker']} {sig['strike']} | {sig['action']} | Edge: +{sig['edge']:.1f}%")
        with bc2:
            if st.button("✅ Mark Traded", key=f"scan_traded_{index}", use_container_width=True):
                st.success("Marked as traded!")

    def _run_scan(self):
        """Execute a full scan."""
        with st.spinner("🔍 Scanning Kalshi markets across all assets..."):
            raw = self.scanner.scan_all_markets()
            signals = self.signal_gen.generate_signals(
                raw, bankroll=st.session_state.scanner_bankroll
            )
            st.session_state.scanner_signals = signals
            st.session_state.scanner_last_scan = datetime.now()
            st.session_state.scanner_count += 1
            total_markets = sum(len(v['markets']) for v in raw.values())
            st.success(f"✅ Found {len(signals)} opportunities from {total_markets} markets")
