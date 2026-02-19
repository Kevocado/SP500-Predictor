"""
Background Scanner — Compute-on-Write
Runs ONCE from GitHub Actions (or locally). Fetches data, runs ML models,
saves a Full Snapshot to Azure Blob (backtesting) and high-conviction
opportunities to Azure Table (UI).

Structured for future expansion: Weather ML, FRED ML, etc.
"""

import os
import sys
import re
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from azure.data.tables import TableClient
from azure.storage.blob import BlobServiceClient

from src.data_loader import fetch_data
from src.feature_engineering import create_features
from src.model import (
    load_model,
    predict_next_hour,
    calculate_probability,
    get_market_volatility,
    kelly_criterion,
)
from src.kalshi_feed import get_real_kalshi_markets
import pandas as pd

# ─── Environment ─────────────────────────────────────────────────────
load_dotenv()
CONN_STR = os.getenv("AZURE_CONNECTION_STRING")

if not CONN_STR:
    print("❌ AZURE_CONNECTION_STRING not set. Exiting.")
    sys.exit(1)

# Edge threshold — matches existing scan_quant() logic (abs(edge) > 5)
EDGE_THRESHOLD = 5.0


# ═════════════════════════════════════════════════════════════════════
# 1. QUANT ML SCANNER  (Heavy — runs in background)
# ═════════════════════════════════════════════════════════════════════

def scan_quant_ml():
    """
    Mirrors HybridScanner.scan_quant() but decoupled from Streamlit.
    Returns (snapshot_records, ui_opportunities).
    """
    tickers = ["SPX", "Nasdaq", "BTC", "ETH"]
    snapshot_records = []
    ui_opportunities = []

    # ── Fetch price data + model predictions (once per asset) ──
    data_cache = {}
    for ticker in tickers:
        try:
            df = fetch_data(ticker, period="5d", interval="1h")
            model, needs_retrain = load_model(ticker)

            if model and not df.empty:
                df_feat = create_features(df)
                pred_val = predict_next_hour(model, df_feat, ticker)
                curr_price = df['Close'].iloc[-1]
                vol = get_market_volatility(df, window=24)

                data_cache[ticker] = {
                    "df": df,
                    "model": model,
                    "vol": vol,
                    "price": curr_price,
                    "pred": pred_val,
                }
                print(f"  ✅ {ticker}: Price={curr_price:.2f}, Pred={pred_val:.2f}, Vol={vol:.6f}")
        except Exception as e:
            print(f"  ⚠️ Skipping {ticker}: {e}")

    # ── Fetch Kalshi markets per ticker and analyze ──
    for ticker in tickers:
        if ticker not in data_cache:
            continue

        d = data_cache[ticker]
        markets, method, debug = get_real_kalshi_markets(ticker)
        print(f"  📡 {ticker}: {len(markets)} markets via {method}")

        for m in markets:
            # Parse strike from title (same regex as scan_quant)
            try:
                strike_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', m.get('title', ''))
                if not strike_match:
                    continue
                strike = float(strike_match.group(1).replace(',', ''))
            except Exception:
                continue

            # Hours left until expiration
            hours_left = 1.0
            exp_str = m.get('expiration')
            if exp_str:
                try:
                    exp = pd.to_datetime(exp_str)
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    hours_left = max(0.1, (exp - datetime.now(timezone.utc)).total_seconds() / 3600)
                except Exception:
                    pass

            # Direction
            title = m.get('title', '')
            is_above = ">" in title or "above" in title.lower()

            # Probability via Black-Scholes + ML drift
            my_prob = calculate_probability(d['price'], d['pred'], strike, d['vol'], hours_left)
            if not is_above:
                my_prob = 100 - my_prob

            # Market price (Kalshi uses cents 0-100)
            yes_ask = m.get('yes_ask', 0)
            edge = my_prob - yes_ask

            # Kelly sizing
            bet_size = kelly_criterion(my_prob, yes_ask, bankroll=20, fractional=0.25)

            # ── Snapshot record (ALWAYS saved — avoids survivorship bias) ──
            record = {
                "ticker": ticker,
                "market_title": title,
                "market_id": m.get('market_id', ''),
                "strike": strike,
                "expiration": exp_str,
                "current_price": round(d['price'], 2),
                "model_pred": round(d['pred'], 2),
                "volatility": round(d['vol'], 6),
                "hours_left": round(hours_left, 1),
                "model_prob": round(my_prob, 2),
                "market_yes_ask": yes_ask,
                "calculated_edge": round(edge, 2),
                "kelly_bet": round(bet_size, 2),
            }
            snapshot_records.append(record)

            # ── UI opportunity (only high conviction) ──
            if abs(edge) > EDGE_THRESHOLD and bet_size > 0:
                action = "BUY YES" if edge > 0 else "BUY NO"
                ui_op = {
                    "PartitionKey": "Live",
                    "RowKey": f"{ticker}_{m.get('market_id', strike)}".replace(" ", ""),
                    "Asset": ticker,
                    "Market": title[:200],
                    "Strike": str(strike),
                    "Confidence": float(round(my_prob, 1)),
                    "Edge": float(round(edge, 1)),
                    "Action": action,
                    "KellySuggestion": float(round(bet_size, 2)),
                    "CurrentPrice": float(round(d['price'], 2)),
                    "ModelPred": float(round(d['pred'], 2)),
                    "Volatility": float(round(d['vol'], 6)),
                    "HoursLeft": float(round(hours_left, 1)),
                    "MarketYesAsk": int(yes_ask),
                    "Expiration": exp_str or "",
                    "MarketId": m.get('market_id', ''),
                }
                ui_opportunities.append(ui_op)

    return snapshot_records, ui_opportunities


# ═════════════════════════════════════════════════════════════════════
# 2. FUTURE: Weather ML  (uncomment when weather model is trained)
# ═════════════════════════════════════════════════════════════════════
# def scan_weather_ml():
#     """Move weather arb to background when ML is added."""
#     pass

# ═════════════════════════════════════════════════════════════════════
# 3. FUTURE: Macro/FRED ML
# ═════════════════════════════════════════════════════════════════════
# def scan_macro_models():
#     """Move FRED analysis to background when ML is added."""
#     pass


# ═════════════════════════════════════════════════════════════════════
# MAIN — Orchestrator
# ═════════════════════════════════════════════════════════════════════

def run_snapshot_logic():
    """Run all background scans, save to Azure Blob + Table."""
    now = datetime.now(timezone.utc)
    print(f"🚀 Starting Background Scan at {now.isoformat()}")

    # ── Initialize Azure clients ──
    blob_service = BlobServiceClient.from_connection_string(CONN_STR)
    table_client = TableClient.from_connection_string(CONN_STR, "CurrentOpportunities")

    # Create containers/tables if missing
    try:
        blob_service.create_container("market-snapshots")
    except Exception:
        pass
    try:
        table_client.create_table()
    except Exception:
        pass

    # ── Run scans ──
    print("\n🧠 Running Quant ML...")
    snapshot_records, ui_opportunities = scan_quant_ml()

    # Future expansion:
    # print("⛈️ Running Weather ML...")
    # weather_snap, weather_ui = scan_weather_ml()
    # snapshot_records.extend(weather_snap)
    # ui_opportunities.extend(weather_ui)

    # ── Save Full Snapshot to Blob (Historical / Backtesting) ──
    full_snapshot = {
        "timestamp_utc": now.isoformat(),
        "markets_analyzed": len(snapshot_records),
        "opportunities_found": len(ui_opportunities),
        "records": snapshot_records,
    }
    blob_name = f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"
    blob_client = blob_service.get_blob_client(container="market-snapshots", blob=blob_name)
    blob_client.upload_blob(json.dumps(full_snapshot, default=str), overwrite=True)
    print(f"✅ Saved full snapshot: {blob_name} ({len(snapshot_records)} markets)")

    # ── Update Azure Table for UI (Live View) ──
    # Clear stale "Live" entries
    try:
        entities = table_client.query_entities("PartitionKey eq 'Live'")
        for e in entities:
            table_client.delete_entity("Live", e['RowKey'])
    except Exception:
        pass

    # Upload new opportunities
    for op in ui_opportunities:
        try:
            table_client.create_entity(op)
        except Exception as e:
            print(f"  ⚠️ Failed to upsert {op.get('RowKey')}: {e}")

    print(f"✅ Updated Azure Table with {len(ui_opportunities)} high-conviction opportunities")
    print(f"🏁 Scan complete at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    run_snapshot_logic()
