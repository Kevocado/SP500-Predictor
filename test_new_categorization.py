#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from kalshi_feed import get_real_kalshi_markets

# Import the real categorization from streamlit_app
sys.path.insert(0, '.')
from streamlit_app import categorize_markets

for ticker in [' BTC', 'ETH', 'SPX']:
    print(f"\n{'='*60}")
    print(f"{ticker} Categorization Test")
    print(f"{'='*60}")
    
    markets = get_real_kalshi_markets(ticker)
    buckets = categorize_markets(markets, ticker)
    
    print(f"\nResults:")
    print(f"  Hourly: {len(buckets['hourly'])}")
    print(f"  Daily: {len(buckets['daily'])}")
    print(f"  Range: {len(buckets['range'])}")
    
    if buckets['daily']:
        m = buckets['daily'][0]
        print(f"\n  First daily market:")
        print(f"    Type: {m.get('market_type')}")
        print(f"    Strike: {m.get('strike_price')}")
        print(f"    Subtitle: {m.get('yes_subtitle', 'N/A')}")
