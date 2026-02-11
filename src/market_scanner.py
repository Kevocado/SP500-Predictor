"""
Hybrid Scanner — Category-Aware Market Scanner
Combines ML predictions, smart money detection, weather scanning,
arbitrage, and yield farming across the full Kalshi catalog.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.kalshi_feed import get_all_active_markets

# Import ML pipeline (graceful fallback if not available)
try:
    from src.data_loader import fetch_data
    from src.feature_engineering import create_features
    from src.model import load_model, predict_next_hour
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML Modules not found. 'AI Predictor' will be disabled.")


class HybridScanner:
    def __init__(self):
        self.markets = []
        self.last_scan = None

    def run_scan(self):
        """
        Executes the Hybrid Scan across all strategies.
        Returns dict with categorized results.
        """
        self.markets = get_all_active_markets(limit_pages=10)
        self.last_scan = datetime.now(timezone.utc)

        # Count categories for logging
        cats = {}
        for m in self.markets:
            cats[m['category']] = cats.get(m['category'], 0) + 1
        print(f"📊 Category breakdown: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")

        return {
            "ml_alpha": self._scan_financial_ml(),
            "smart_money": self._scan_smart_money(),
            "weather": self._scan_weather(),
            "arbitrage": self._find_arbitrage(),
            "yield_farming": self._find_yield_farms()
        }

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 1: ML Alpha — SPX/BTC/ETH/Nasdaq LightGBM predictions
    # ═══════════════════════════════════════════════════════════════
    def _scan_financial_ml(self):
        """
        Scans SPX, BTC, ETH, Nasdaq using trained LightGBM models.
        Matches predictions against live Kalshi markets.
        """
        if not ML_AVAILABLE:
            return []

        signals = []
        targets = ["SPX", "BTC", "ETH", "Nasdaq"]

        # Filter to financial markets only
        fin_markets = [m for m in self.markets
                       if m['category'] in ['Financials', 'Economics']]

        for ticker in targets:
            try:
                # 1. Fetch Data & Model
                df = fetch_data(ticker, period="5d", interval="1m")
                model, _ = load_model(ticker)

                if model and not df.empty:
                    # 2. Predict
                    df_feat = create_features(df)
                    pred_val = predict_next_hour(model, df_feat, ticker)
                    curr_price = df['Close'].iloc[-1]

                    # 3. Match with Kalshi Markets
                    for m in fin_markets:
                        # Match by ticker pattern
                        is_match = False
                        if ticker == "SPX" and "INX" in m.get('ticker', ''):
                            is_match = True
                        elif ticker == "Nasdaq" and "NAS" in m.get('ticker', ''):
                            is_match = True
                        elif ticker in m.get('ticker', '') or ticker in m.get('title', ''):
                            is_match = True

                        if not is_match:
                            continue

                        # Parse strike from title
                        try:
                            strike_match = re.search(
                                r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
                                m['title']
                            )
                            if not strike_match:
                                continue
                            strike = float(strike_match.group(1).replace(',', ''))
                        except:
                            continue

                        # Determine direction
                        direction = "Bullish" if (">" in m['title'] or "above" in m['title'].lower()) else "Bearish"

                        # Generate signal
                        if direction == "Bullish" and pred_val > strike:
                            conf = (pred_val - strike) / strike * 100
                            signals.append({
                                "Asset": ticker,
                                "Market": m['title'],
                                "Model_Pred": pred_val,
                                "Strike": strike,
                                "Kalshi_Price": m['price'],
                                "Action": "BUY YES",
                                "Edge": "Bullish",
                                "Confidence": f"{conf:.2f}%"
                            })
                        elif direction == "Bearish" and pred_val < strike:
                            signals.append({
                                "Asset": ticker,
                                "Market": m['title'],
                                "Model_Pred": pred_val,
                                "Strike": strike,
                                "Kalshi_Price": m['price'],
                                "Action": "BUY YES",
                                "Edge": "Bearish",
                                "Confidence": "High"
                            })

            except Exception as e:
                print(f"   ⚠️ Skipping ML for {ticker}: {e}")
                continue

        return signals

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 2: Smart Money — Economics & Politics
    # ═══════════════════════════════════════════════════════════════
    def _scan_smart_money(self):
        """Filters for Economics and Politics (The 'Real' Markets)."""
        return [m for m in self.markets
                if m['category'] in ['Economics', 'Politics']]

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 3: Weather Markets
    # ═══════════════════════════════════════════════════════════════
    def _scan_weather(self):
        """Filters for Climate & Weather markets."""
        return [m for m in self.markets
                if m['category'] == 'Weather']

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 4: Arbitrage — Free Money across ALL categories
    # ═══════════════════════════════════════════════════════════════
    def _find_arbitrage(self):
        """Finds negative spreads (Free Money) across ALL categories."""
        opps = []
        for m in self.markets:
            if m['no_price'] > 0:
                cost = m['price'] + m['no_price']
                if cost < 100:
                    opps.append({
                        **m,
                        "cost": cost,
                        "profit": 100 - cost
                    })
        opps.sort(key=lambda x: x['profit'], reverse=True)
        return opps

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 5: Yield Farming — Safe bets, excludes Sports
    # ═══════════════════════════════════════════════════════════════
    def _find_yield_farms(self):
        """Finds safe 92-98¢ bets (<48h) excluding Sports."""
        opps = []
        now = datetime.now(timezone.utc)

        for m in self.markets:
            if m['category'] == 'Sports':
                continue  # Skip sports for safe yield

            if 92 <= m['price'] <= 98:
                if not m['expiration']:
                    continue
                try:
                    exp = pd.to_datetime(m['expiration'])
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    hours = (exp - now).total_seconds() / 3600

                    if 0 < hours < 48:
                        roi = (100 - m['price']) / m['price'] * 100
                        opps.append({
                            **m,
                            "hours_left": int(hours),
                            "roi": roi
                        })
                except:
                    continue

        opps.sort(key=lambda x: x['roi'], reverse=True)
        return opps
