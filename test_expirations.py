import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

# Load .env
root_dir = Path(__file__).parent
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

def test_market_expirations():
    """Test script to see actual expiration times of Kalshi markets"""
    
    tickers = {
        'BTC': 'KXBTC',
        'ETH': 'KXETH',
        'SPX': 'KXINX',
        'Nasdaq': 'KXNASDAQ100'
    }
    
    now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    ny_tz = ZoneInfo("America/New_York")
    now_ny = now_utc.astimezone(ny_tz)
    
    print(f"Current time (UTC): {now_utc}")
    print(f"Current time (NY):  {now_ny}")
    print(f"=" * 80)
    
    for ticker, series in tickers.items():
        print(f"\n{ticker} ({series}):")
        print("-" * 80)
        
        params = {
            "series_ticker": series,
            "status": "open",
            "limit": 100
        }
        
        response = requests.get(KALSHI_API_URL, params=params)
        if response.status_code == 200:
            markets = response.json().get('markets', [])
            print(f"Total markets: {len(markets)}")
            
            # Categorize
            hourly = []
            daily = []
            future = []
            
            for m in markets[:10]:  # Check first 10
                exp_str = m.get('expiration_time')
                if not exp_str:
                    continue
                    
                exp_time = pd.to_datetime(exp_str)
                if exp_time.tzinfo is None:
                    exp_time = exp_time.replace(tzinfo=ZoneInfo("UTC"))
                
                exp_ny = exp_time.astimezone(ny_tz)
                time_diff_min = (exp_time - now_utc).total_seconds() / 60.0
                
                strike = m.get('strike_price', 'N/A')
                title = m.get('title', '')[:50]
                
                if 0 < time_diff_min <= 90:
                    hourly.append((strike, exp_ny, time_diff_min))
                    print(f"  ⚡ HOURLY: Strike={strike}, Exp={exp_ny.strftime('%m/%d %H:%M')}, In {time_diff_min:.0f}min")
                elif exp_ny.date() == now_ny.date() and 15 <= exp_ny.hour <= 23:
                    daily.append((strike, exp_ny, time_diff_min))
                    print(f"  📅 DAILY:  Strike={strike}, Exp={exp_ny.strftime('%m/%d %H:%M')}, In {time_diff_min:.0f}min")
                else:
                    future.append((strike, exp_ny, time_diff_min))
                    print(f"  🔮 FUTURE: Strike={strike}, Exp={exp_ny.strftime('%m/%d %H:%M')}, In {time_diff_min:.0f}min ({time_diff_min/60:.1f}h)")
            
            print(f"\nSummary: {len(hourly)} hourly, {len(daily)} daily, {len(future)} future")
        else:
            print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_market_expirations()
