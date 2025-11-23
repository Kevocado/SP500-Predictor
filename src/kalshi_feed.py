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
    Returns a tuple: (list of market dicts, fetch_method_string)
    """
    if not API_KEY:
        print("ℹ️ KALSHI_API_KEY not found. Attempting public data fetch...")
    
    # Map our tickers to Kalshi's actual series tickers
    ticker_map = {
        "BTC": "KXBTC",           # Bitcoin
        "ETH": "KXETH",           # Ethereum
        "SPX": "KXINX",           # S&P 500
        "Nasdaq": "KXNASDAQ100"   # Nasdaq-100
    }
    
    series_ticker = ticker_map.get(ticker)
    if not series_ticker:
        print(f"⚠️ No Kalshi series mapping for {ticker}")
        return [], "Error: No Mapping"

    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    # --- STEP A: Targeted Fetch (Precise) ---
    try:
        print(f"🔍 [Step A] Targeted Fetch for {ticker} (Series: {series_ticker})...")
        params_targeted = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 1000
        }
        
        response = requests.get(
            KALSHI_API_URL, 
            params=params_targeted, 
            headers=headers if API_KEY else None,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
            if markets:
                print(f"   ✅ Targeted fetch success: {len(markets)} markets found.")
                return process_markets(markets, ticker), "Targeted"
            else:
                print("   ⚠️ Targeted fetch returned 0 markets. Proceeding to fallback...")
        else:
            print(f"   ❌ Targeted fetch failed: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Targeted fetch error: {e}")

    # --- STEP B: Fallback Fetch (Broad) ---
    try:
        print(f"🔍 [Step B] Fallback Fetch (Broad Search) for {ticker}...")
        params_fallback = {
            "limit": 1000,
            "status": "open"
        }
        
        fb_response = requests.get(
            KALSHI_API_URL, 
            params=params_fallback, 
            headers=headers if API_KEY else None,
            timeout=15
        )
        
        if fb_response.status_code == 200:
            fb_data = fb_response.json()
            all_markets = fb_data.get('markets', [])
            print(f"   Broad fetch got {len(all_markets)} total markets. Filtering client-side...")
            
            # Client-Side Filter: Keep if ticker symbol is in the market ticker string
            # e.g. "KXBTC" in "KXBTC-23NOV..." OR "BTC" in "KXBTC..."
            filtered_markets = []
            for m in all_markets:
                m_ticker = m.get('ticker', '')
                if series_ticker in m_ticker or ticker in m_ticker:
                    filtered_markets.append(m)
            
            if filtered_markets:
                print(f"   ✅ Fallback success: {len(filtered_markets)} markets matched.")
                return process_markets(filtered_markets, ticker), "Fallback (Broad)"
            else:
                print("   ❌ Fallback found 0 matching markets.")
                return [], "Failed (0 Found)"
        else:
            print(f"   ❌ Fallback fetch failed: {fb_response.status_code}")
            return [], "Failed (API Error)"

    except Exception as e:
        print(f"   ❌ Fallback fetch error: {e}")
        return [], "Failed (Exception)"

def process_markets(markets, ticker):
    """Helper to process raw market data into our format."""
    results = []
    for m in markets:
        floor = m.get('floor_strike')
        cap = m.get('cap_strike')
        
        is_range = (floor is not None and cap is not None)
        
        if is_range:
            strike_price = None
            market_type = 'range'
        elif floor is not None:
            strike_price = floor
            market_type = 'above'
        elif cap is not None:
            strike_price = cap
            market_type = 'below'
        else:
            continue
        
        results.append({
            'ticker': ticker,
            'strike_price': strike_price,
            'floor_strike': floor,
            'cap_strike': cap,
            'market_type': market_type,
            'yes_bid': m.get('yes_bid', 0),
            'no_bid': m.get('no_bid', 0),
            'yes_ask': m.get('yes_ask', 0),
            'no_ask': m.get('no_ask', 0),
            'expiration': m.get('expiration_time'),
            'market_id': m.get('ticker'),
            'title': m.get('title', ''),
            'yes_subtitle': m.get('yes_sub_title', ''),
            'no_subtitle': m.get('no_sub_title', '')
        })
    return results

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
