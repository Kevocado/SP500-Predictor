import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kalshi_feed import get_real_kalshi_markets, check_kalshi_connection
import json

print("--- Checking Connection ---")
is_connected = check_kalshi_connection()
print(f"Connection Status: {is_connected}")

print("\n--- Fetching SPX Markets ---")
markets, method, debug_info = get_real_kalshi_markets("SPX")

print(f"\nMethod: {method}")
print(f"Market Count: {len(markets)}")
print("\nDebug Info:")
print(json.dumps(debug_info, indent=2, default=str))

if markets:
    print("\nSample Market:")
    print(markets[0])
else:
    print("\nNo OPEN markets found. Checking CLOSED markets to verify ticker...")
    import requests
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    
    # Check KXINX specifically without status filter
    print("\n--- Checking KXINX (Any Status) ---")
    resp = requests.get(url, params={"series_ticker": "KXINX", "limit": 5})
    if resp.status_code == 200:
        found = resp.json().get('markets', [])
        print(f"Found {len(found)} markets for KXINX (any status).")
        if found:
            print(f"Sample: {found[0]['ticker']} | Status: {found[0]['status']}")
    else:
        print(f"Error checking KXINX: {resp.status_code}")
    print("\n--- Raw Market Analysis ---")
    import requests
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    # Fetch 500 markets
    resp = requests.get(url, params={"limit": 500, "status": "open"})
    if resp.status_code == 200:
        raw = resp.json().get('markets', [])
        print(f"Fetched {len(raw)} raw markets.")
        
        print("\nSearching for 'S&P', '500', 'Nasdaq' in TITLES:")
        matches = []
        for m in raw:
            title = m.get('title', '').lower()
            if 's&p' in title or '500' in title or 'nasdaq' in title:
                matches.append(f"{m.get('ticker')} | {m.get('title')}")
        
        for m in matches[:10]:
            print(m)
            
        if not matches:
            print("No S&P/Nasdaq matches found in top 500.")

print("\n--- Checking BTC ---")
# Check if BTC works
markets_btc, _, _ = get_real_kalshi_markets("BTC")
print(f"BTC Markets Found: {len(markets_btc)}")
if markets_btc:
    print(markets_btc[0]['market_id'])
