import os
import requests
from datetime import datetime
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

def fetch_macro_data():
    """
    Fetches the latest US Core CPI and 10-Year Treasury Yields from FRED.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY missing from environment.")
        return None

    try:
        fred = Fred(api_key=api_key)

        # CPILFESL is Consumer Price Index for All Urban Consumers: All Items Less Food and Energy (Core CPI)
        core_cpi_series = fred.get_series('CPILFESL')
        latest_cpi = core_cpi_series.iloc[-1]
        prev_cpi = core_cpi_series.iloc[-2]

        # Calculate MoM change
        mom_cpi_change = ((latest_cpi - prev_cpi) / prev_cpi) * 100

        # DGS10 is Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
        ten_year_yield_series = fred.get_series('DGS10')
        # DGS10 occasionally has NaNs (weekends/holidays), get the last valid value
        latest_10y_yield = ten_year_yield_series.dropna().iloc[-1]

        return {
            "core_cpi_mom": round(mom_cpi_change, 2),
            "core_cpi_level": latest_cpi,
            "10y_yield": latest_10y_yield
        }
    except Exception as e:
        print(f"Error fetching data from FRED: {e}")
        return None

def fetch_kalshi_market_data(ticker):
    """
    Fetches market pricing from Kalshi for a given ticker.
    """
    api_key = os.getenv("KALSHI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        r = requests.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook",
            headers=headers, timeout=5
        )
        if r.status_code != 200:
            return None

        data = r.json().get('orderbook', {})
        yes_bids = data.get('yes', []) or []
        no_bids = data.get('no', []) or []

        best_yes = yes_bids[-1][0] if yes_bids else 0
        best_no = no_bids[-1][0] if no_bids else 0

        if best_yes == 0 and best_no == 0:
            return None

        # Implied probability is roughly the yes price / 100
        implied_prob = best_yes / 100.0 if best_yes > 0 else (100 - best_no) / 100.0

        return {
            "best_yes_cents": best_yes,
            "best_no_cents": best_no,
            "implied_prob": implied_prob
        }
    except Exception as e:
        print(f"Error fetching Kalshi data for {ticker}: {e}")
        return None

def calculate_macro_edge():
    """
    Compares real macroeconomic data against Kalshi markets to find > 5% edge.
    """
    macro_data = fetch_macro_data()
    if not macro_data:
        return []

    print(f"Latest FRED Data: Core CPI MoM {macro_data['core_cpi_mom']}%, 10Y Yield {macro_data['10y_yield']}%")

    opportunities = []

    # ---------------------------------------------------------
    # Example 1: Core CPI Market
    # In a real scenario, you'd map to the active Kalshi CPI market ticker.
    # We will simulate a market checking if CPI is > 0.2%
    # ---------------------------------------------------------
    cpi_market_ticker = "KXCPI-24DEC-T0.2" # Placeholder ticker format for CPI > 0.2%

    # Let's say our math model says: if the actual trailing CPI is already higher
    # than the market expectations, the real probability is very high.
    # Here we are just using a simple heuristic for the sake of the engine.
    our_cpi_prob = 0.90 if macro_data['core_cpi_mom'] > 0.2 else 0.10

    # In a real scenario you would fetch the actual market ticker.
    # For testing, we mock a response if the market isn't found.
    market_cpi_data = fetch_kalshi_market_data(cpi_market_ticker)

    if market_cpi_data:
        market_cpi_prob = market_cpi_data["implied_prob"]
        edge = our_cpi_prob - market_cpi_prob

        if abs(edge) > 0.05: # > 5% edge
            opportunities.append({
                "PartitionKey": "MACRO_ENGINE",
                "RowKey": f"CPI_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "MarketTicker": cpi_market_ticker,
                "SignalType": "Core CPI",
                "CalculatedProb": our_cpi_prob,
                "MarketProb": market_cpi_prob,
                "Edge": round(edge, 4),
                "Action": "BUY YES" if edge > 0 else "BUY NO",
                "Reasoning": f"Calculated MoM CPI is {macro_data['core_cpi_mom']}%, our prob is {our_cpi_prob:.2f} but market prices at {market_cpi_prob:.2f}",
                "Timestamp": datetime.utcnow().isoformat()
            })

    # ---------------------------------------------------------
    # Example 2: 10-Year Yield Market
    # ---------------------------------------------------------
    yield_market_ticker = "KXYIELD-24DEC-T4.0" # Placeholder for 10Y > 4.0%

    our_yield_prob = 0.85 if macro_data['10y_yield'] > 4.0 else 0.15
    market_yield_data = fetch_kalshi_market_data(yield_market_ticker)

    if market_yield_data:
        market_yield_prob = market_yield_data["implied_prob"]
        edge = our_yield_prob - market_yield_prob

        if abs(edge) > 0.05:
            opportunities.append({
                "PartitionKey": "MACRO_ENGINE",
                "RowKey": f"YIELD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "MarketTicker": yield_market_ticker,
                "SignalType": "10Y Treasury Yield",
                "CalculatedProb": our_yield_prob,
                "MarketProb": market_yield_prob,
                "Edge": round(edge, 4),
                "Action": "BUY YES" if edge > 0 else "BUY NO",
                "Reasoning": f"10Y Yield is {macro_data['10y_yield']}%, our prob is {our_yield_prob:.2f} but market prices at {market_yield_prob:.2f}",
                "Timestamp": datetime.utcnow().isoformat()
            })

    return opportunities

if __name__ == "__main__":
    print("Running Macro Engine...")
    ops = calculate_macro_edge()
    if not ops:
        print("No macro opportunities with > 5% edge found (or unable to fetch market data).")
    for op in ops:
        print(op)