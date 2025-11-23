#!/usr/bin/env python3
"""Test market categorization logic with new market_type field."""

import sys
sys.path.insert(0, 'src')

from kalshi_feed import get_real_kalshi_markets
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

def categorize_markets(markets, ticker):
    """
    Copy of the UPDATED categorization logic from streamlit_app.py
    """
    from zoneinfo import ZoneInfo

    buckets = {'hourly': [], 'daily': [], 'range': []}

    now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    ny_tz = ZoneInfo("America/New_York")
    now_ny = now_utc.astimezone(ny_tz)

    is_crypto = ticker in ["BTC", "ETH"]

    for m in markets:
        try:
            exp_str = m.get('expiration')
            if not exp_str:
                continue

            # Parse expiration preserving timezone info if present
            exp_time = pd.to_datetime(exp_str)
            if exp_time.tzinfo is None:
                # assume UTC if no tz provided
                exp_time = exp_time.replace(tzinfo=ZoneInfo("UTC"))

            # Normalize to NY timezone for date/hour checks
            exp_ny = exp_time.astimezone(ny_tz)

            # Range detection - use market_type field from Kalshi API
            # market_type can be: 'above', 'below', or 'range'
            market_type = m.get('market_type', '')
            
            if market_type == 'range':
                buckets['range'].append(m)
                continue

            # Time difference in minutes from now (UTC-based compare)
            time_diff_min = (exp_time - now_utc).total_seconds() / 60.0
            time_diff_hours = time_diff_min / 60.0
            time_diff_days = time_diff_hours / 24.0

            # Hourly: expires within 90 minutes AND (for crypto) inside allowed NY hours
            if 0 < time_diff_min <= 90:
                if is_crypto:
                    # Crypto active window: 09:00 - 23:59 NY time
                    if 9 <= now_ny.hour <= 23:
                        buckets['hourly'].append(m)
                else:
                    buckets['hourly'].append(m)
                continue

            # Daily: expires within 14 days (more flexible to show available markets)
            # Extended to 14 days to catch markets that expire a week+ out
            if 0 < time_diff_days <= 14:
                buckets['daily'].append(m)
                continue

        except Exception as e:
            print(f"Error categorizing market: {e}")
    
    print(f"📊 Categorization for {ticker}:")
    print(f"   Hourly: {len(buckets['hourly'])} markets")
    print(f"   Daily: {len(buckets['daily'])} markets")
    print(f"   Range: {len(buckets['range'])} markets")
    print(f"   Current NY time: {now_ny.strftime('%Y-%m-%d %H:%M %Z')}")

    return buckets

# Test each ticker
for ticker in ['SPX', 'BTC', 'ETH']:
    print(f"\n{'='*60}")
    print(f"Testing {ticker}")
    print(f"{'='*60}")
    
    markets = get_real_kalshi_markets(ticker)
    print(f"Fetched {len(markets)} total markets")
    
    buckets = categorize_markets(markets, ticker)
    
    # Show samples
    if buckets['daily']:
        print(f"\n  Sample daily market:")
        m = buckets['daily'][0]
        print(f"    Type: {m.get('market_type')}")
        print(f"    Strike: {m.get('strike_price')}")
        print(f"    Subtitle: {m.get('yes_subtitle', 'N/A')}")
        print(f"    Exp: {m.get('expiration')}")
    
    if buckets['range']:
        print(f"\n  Sample range market:")
        m = buckets['range'][0]
        print(f"    Type: {m.get('market_type')}")
        print(f"    Subtitle: {m.get('yes_subtitle', 'N/A')}")
