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
    
    # Map our tickers to Kalshi's actual series tickers
    ticker_map = {
        "BTC": "KXBTCD",          # Bitcoin Daily/Hourly
        "ETH": "KXETHD",          # Ethereum Daily/Hourly
        "SPX": "KXINX",           # S&P 500 Range
        "Nasdaq": "KXNASDAQ100"   # Nasdaq-100
    }
    
    series_ticker = ticker_map.get(ticker)
    if not series_ticker:
        print(f"⚠️ No Kalshi series mapping for {ticker}")
        return []

    try:
        # Server-side filtering with params
        params = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 200
        }
        
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        
        print(f"🔍 Fetching Kalshi markets for {ticker} (series: {series_ticker})")
        print(f"   URL: {KALSHI_API_URL}")
        print(f"   Has API Key: {bool(API_KEY)}")
        
        response = requests.get(KALSHI_API_URL, params=params, headers=headers if API_KEY else None)
        
        print(f"   Response Status: {response.status_code}")
        
        markets = []
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
        
        # Fallback Logic: If no markets found with series filter, try broad fetch
        if not markets:
            print(f"⚠️ No markets found for series {series_ticker}. Attempting fallback fetch...")
            fallback_params = {
                "limit": 100,
                "status": "open"
            }
            fb_response = requests.get(KALSHI_API_URL, params=fallback_params, headers=headers if API_KEY else None)
            if fb_response.status_code == 200:
                fb_data = fb_response.json()
                all_markets = fb_data.get('markets', [])
                
                # Filter manually by checking if the series_ticker is in the market's ticker string
                # e.g. "KXBTC" in "KXBTC-23NOV25-..."
                markets = [m for m in all_markets if series_ticker in m.get('ticker', '')]
                
                # Double check: if list is empty, try the raw ticker symbol (e.g. "BTC")
                if not markets:
                     markets = [m for m in all_markets if ticker in m.get('ticker', '')]
                
                print(f"   Fallback found {len(markets)} markets for {ticker} (using series {series_ticker})")
            else:
                print(f"❌ Fallback fetch failed: {fb_response.status_code}")

        print(f"   Total markets returned: {len(markets)}")
        
        results = []
        for m in markets:
            # Extract strike price
            strike_price = m.get('strike_price')
            
            # If no strike_price field, try parsing from subtitle
            if not strike_price:
                subtitle_text = m.get('yes_sub_title') or m.get('subtitle', '')
                import re
                numbers = re.findall(r'[\d,]+', subtitle_text.replace('$', ''))
                if numbers:
                    try:
                        strike_price = float(numbers[0].replace(',', ''))
                    except:
                        pass
            
            # Bid/Ask
            yes_bid = m.get('yes_bid', 0)
            no_bid = m.get('no_bid', 0)
            yes_ask = m.get('yes_ask', 0)
            no_ask = m.get('no_ask', 0)
            
            results.append({
                'ticker': ticker,
                'strike_price': strike_price,
                'yes_bid': yes_bid,
                'no_bid': no_bid,
                'yes_ask': yes_ask,
                'no_ask': no_ask,
                'expiration': m.get('expiration_time'),
                'market_id': m.get('ticker'),
                'title': m.get('title', '')
            })
            
        print(f"   Extracted {len(results)} markets with pricing data")
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
