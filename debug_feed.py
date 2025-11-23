
import sys
import os
sys.path.append(os.getcwd())
from src.kalshi_feed import get_real_kalshi_markets
from datetime import datetime, timedelta
import pandas as pd

def test_feed():
    tickers = ["BTC", "ETH", "SPX", "Nasdaq"]
    for ticker in tickers:
        print(f"--- Checking {ticker} ---")
        markets = get_real_kalshi_markets(ticker)
        print(f"Found {len(markets)} markets.")
        if markets:
            print("Sample market FULL JSON:", markets[0])
            
            # Test categorization logic
            now = datetime.utcnow()
            offset = timedelta(hours=-5)
            now_et = now + offset
            print(f"Current UTC: {now}, ET: {now_et}")
            
            for m in markets[:5]:
                exp_str = m.get('expiration')
                if not exp_str: continue
                exp_time = pd.to_datetime(exp_str).replace(tzinfo=None)
                time_diff = (exp_time - now).total_seconds() / 60
                
                print(f"Exp: {exp_time}, Diff: {time_diff:.1f} min")
                
                if 0 < time_diff <= 90:
                    if ticker in ["BTC", "ETH"]:
                        if 9 <= now_et.hour <= 23:
                            print("  -> Included (Crypto Window)")
                        else:
                            print(f"  -> Excluded (Crypto Window: Hour {now_et.hour})")
                    else:
                        print("  -> Included (Index)")

if __name__ == "__main__":
    test_feed()
