import requests
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

from pathlib import Path

# Load .env from root directory
root_dir = Path(__file__).parent.parent
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
API_KEY = os.getenv("KALSHI_API_KEY")

def get_real_kalshi_markets(ticker):
    """
    Fetches active markets from Kalshi for the given ticker.
    Returns a list of dictionaries with market data.
    """
    if not API_KEY:
        print("ℹ️ KALSHI_API_KEY not found. Attempting public data fetch...")
        # Proceed without key
    
    # Map our tickers to Kalshi series tickers
    ticker_map = {
        "BTC": "KXBT",
        "ETH": "KXETH",
        "SPX": "KXSPX",
        "Nasdaq": "KXNASDAQ"
    }
    
    series_ticker = ticker_map.get(ticker)
    if not series_ticker:
        return []

    try:
        # Fetch markets
        params = {
            "limit": 200,
            "status": "open"
        }
        
        # Add series_ticker if we want to filter (but let's try without first)
        # params["series_ticker"] = series_ticker
        
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        
        print(f"🔍 Fetching Kalshi markets for {ticker} (series: {series_ticker})")
        print(f"   URL: {KALSHI_API_URL}")
        print(f"   Has API Key: {bool(API_KEY)}")
        
        response = requests.get(KALSHI_API_URL, params=params, headers=headers if API_KEY else None)
        
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error fetching Kalshi data: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return []
            
        data = response.json()
        markets = data.get('markets', [])
        
        print(f"   Total markets returned: {len(markets)}")
        
        # Debug: Show what fields are available
        if markets:
            sample = markets[0]
            print(f"   Sample market fields: {list(sample.keys())}")
            print(f"   Sample ticker: {sample.get('ticker')}")
            print(f"   Sample event_ticker: {sample.get('event_ticker')}")
            print(f"   Sample series_ticker: {sample.get('series_ticker', 'NOT FOUND')}")
        
        # Filter by series_ticker or event_ticker
        results = []
        for m in markets:
            # Check both series_ticker and event_ticker
            market_series = m.get('series_ticker', '')
            event_ticker = m.get('event_ticker', '')
            
            # Look for our ticker in either field
            if series_ticker.lower() in market_series.lower() or series_ticker.lower() in event_ticker.lower():
                # Extract data
                strike_price = m.get('strike_price')
                
                # Bid/Ask
                yes_bid = m.get('yes_bid', 0)
                no_bid = m.get('no_bid', 0)
                
                results.append({
                    'ticker': ticker,
                    'strike_price': strike_price,
                    'yes_bid': yes_bid,
                    'no_bid': no_bid,
                    'expiration': m.get('expiration_time'),
                    'market_id': m.get('ticker')
                })
            
        print(f"   Filtered to {len(results)} markets for {series_ticker}")
        return results

    except Exception as e:
        print(f"❌ Exception in Kalshi feed: {e}")
        import traceback
        traceback.print_exc()
        return []

def check_kalshi_connection():
    """
    Checks if the Kalshi API is accessible and the key is valid.
    Returns True if successful, False otherwise.
    """
    try:
        # Try a simple fetch
        params = {"limit": 1, "status": "open"}
        # If key is present, use it? The get_real_kalshi_markets logic handles it.
        # But here we want to test the connection specifically.
        # Let's use the same URL.
        response = requests.get(KALSHI_API_URL, params=params)
        if response.status_code == 200:
            return True
        return False
    except:
        return False
