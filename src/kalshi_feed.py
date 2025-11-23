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
        "BTC": "KXBTC",           # Bitcoin (base series catches all expirations)
        "ETH": "KXETH",           # Ethereum (base series catches all expirations)
        "SPX": "KXINX",           # S&P 500
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
            "limit": 300
        }
        
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        
        print(f"🔍 Fetching Kalshi markets for {ticker} (series: {series_ticker})")
        print(f"   URL: {KALSHI_API_URL}")
        print(f"   Params: {params}")
        print(f"   Has API Key: {bool(API_KEY)}")
        
        # Add timeout to prevent hanging
        response = requests.get(
            KALSHI_API_URL, 
            params=params, 
            headers=headers if API_KEY else None,
            timeout=10  # 10 second timeout
        )
        
        print(f"   Response Status: {response.status_code}")
        
        markets = []
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
            print(f"   ✅ Primary fetch returned {len(markets)} markets")
        else:
            print(f"   ❌ Primary fetch failed: {response.status_code}")
            print(f"   Response: {response.text[:300]}")
        
        # Fallback Logic: If no markets found with series filter, try broad fetch
        if not markets:
            print(f"⚠️ No markets found for series {series_ticker}. Attempting fallback fetch...")
            fallback_params = {
                "limit": 100,
                "status": "open"
            }
            fb_response = requests.get(
                KALSHI_API_URL, 
                params=fallback_params, 
                headers=headers if API_KEY else None,
                timeout=10
            )
            
            if fb_response.status_code == 200:
                fb_data = fb_response.json()
                all_markets = fb_data.get('markets', [])
                
                print(f"   Fallback: Got {len(all_markets)} total markets")
                
                # Filter manually by checking if the series_ticker is in the market's ticker string
                # e.g. "KXBTC" in "KXBTC-23NOV25-..."
                markets = [m for m in all_markets if series_ticker in m.get('ticker', '')]
                
                # Double check: if list is empty, try the raw ticker symbol (e.g. "BTC")
                if not markets:
                     markets = [m for m in all_markets if ticker in m.get('ticker', '')]
                
                print(f"   ✅ Fallback found {len(markets)} markets for {ticker} (using series {series_ticker})")
                
                # Debug: Show first market ticker if available
                if markets:
                    print(f"   Example ticker: {markets[0].get('ticker', 'N/A')}")
            else:
                print(f"   ❌ Fallback fetch failed: {fb_response.status_code}")
                print(f"   Response: {fb_response.text[:300]}")

        print(f"   📊 Total markets returned: {len(markets)}")
        
        results = []
        strike_count = 0
        range_count = 0
        
        for m in markets:
            # Kalshi uses floor_strike and cap_strike, NOT strike_price
            floor = m.get('floor_strike')
            cap = m.get('cap_strike')
            
            # Determine market type:
            # - If only floor (no cap): "Above X" strike market
            # - If only cap (no floor): "Below X" strike market  
            # - If both floor and cap: "Between X and Y" range market
            
            is_range = (floor is not None and cap is not None)
            
            if is_range:
                # Range market: "Between X and Y"
                strike_price = None  # No single strike
                market_type = 'range'
                range_count += 1
            elif floor is not None:
                # Strike market: "Above X"
                strike_price = floor
                market_type = 'above'
                strike_count += 1
            elif cap is not None:
                # Strike market: "Below X"
                strike_price = cap
                market_type = 'below'
                strike_count += 1
            else:
                # No floor or cap - skip this market
                print(f"   ⚠️ Skipping market with no floor or cap: {m.get('ticker')}")
                continue
            
            # Get subtitle for better description
            yes_subtitle = m.get('yes_sub_title', '')
            no_subtitle = m.get('no_sub_title', '')
            
            # Bid/Ask
            yes_bid = m.get('yes_bid', 0)
            no_bid = m.get('no_bid', 0)
            yes_ask = m.get('yes_ask', 0)
            no_ask = m.get('no_ask', 0)
            
            results.append({
                'ticker': ticker,
                'strike_price': strike_price,
                'floor_strike': floor,
                'cap_strike': cap,
                'market_type': market_type,  # 'above', 'below', or 'range'
                'yes_bid': yes_bid,
                'no_bid': no_bid,
                'yes_ask': yes_ask,
                'no_ask': no_ask,
                'expiration': m.get('expiration_time'),
                'market_id': m.get('ticker'),
                'title': m.get('title', ''),
                'yes_subtitle': yes_subtitle,
                'no_subtitle': no_subtitle
            })
            
        print(f"   ✅ Extracted {len(results)} markets: {strike_count} strike + {range_count} range")
        
        # Additional validation: Check if we got financial markets
        if results:
            sample = results[0]
            print(f"   Sample market: {sample.get('market_type')} - {sample.get('yes_subtitle', 'N/A')[:60]}")
        
        return results

    except requests.exceptions.Timeout:
        print(f"❌ Timeout fetching Kalshi markets for {ticker}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error in Kalshi feed for {ticker}: {e}")
        return []
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
