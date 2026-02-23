import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env
root_dir = Path(__file__).parent.parent
load_dotenv(dotenv_path=root_dir / '.env', override=True)

class MicrostructureEngine:
    """
    PhD Milestone 3: Order Book Microstructure
    Analyzes Kalshi market depth to detect institutional 'whales' and order flow skew.
    """

    def __init__(self):
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.api_key = os.getenv("KALSHI_API_KEY") # This is for signed requests if needed, 
                                                   # but orderbook is public.

    def fetch_order_book(self, ticker):
        """Fetches the full order book depth for a specific market."""
        url = f"{self.base_url}/markets/{ticker}/orderbook"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
            return None
        except Exception:
            return None

    def analyze_skew(self, ticker):
        """
        Calculates the buy/sell imbalance (skew) of the order book.
        - Positive skew: More demand for YES (bullish).
        - Negative skew: More demand for NO (bearish).
        """
        book = self.fetch_order_book(ticker)
        if not book or not book.get('orderbook'):
            return {"skew": 0, "whale_detected": False, "signal": "Neutral"}

        ob = book['orderbook']
        yes_orders = ob.get('yes') or []
        no_orders = ob.get('no') or []

        # Sum of quantity * price to get total dollar depth
        yes_depth = sum(int(o[0]) * int(o[1]) for o in yes_orders)
        no_depth = sum(int(o[0]) * int(o[1]) for o in no_orders)

        total_depth = yes_depth + no_depth
        if total_depth == 0:
            return {"skew": 0, "whale_detected": False, "signal": "Neutral"}

        skew = (yes_depth - no_depth) / total_depth
        
        # Detect 'whales' (single large orders > 1000 contracts at a single price)
        yes_whale = any(int(o[1]) > 1000 for o in yes_orders)
        no_whale = any(int(o[1]) > 1000 for o in no_orders)
        
        signal = "Neutral"
        if skew > 0.3: signal = "Institutional Overweight: YES"
        elif skew < -0.3: signal = "Institutional Overweight: NO"

        return {
            "ticker": ticker,
            "skew": round(skew * 100, 1),
            "yes_depth": yes_depth,
            "no_depth": no_depth,
            "whale_detected": yes_whale or no_whale,
            "whale_side": "YES" if yes_whale else ("NO" if no_whale else "None"),
            "signal": signal
        }

if __name__ == "__main__":
    # Test
    engine = MicrostructureEngine()
    # Replace with a real active ticker if possible, or just print the structure
    ticker = "KXHIGHNY-26FEB22-T45" 
    print(f"Testing skew analysis for {ticker}...")
    print(json.dumps(engine.analyze_skew(ticker), indent=2))
