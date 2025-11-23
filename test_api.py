import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load .env from root directory explicitly
root_dir = Path(__file__).parent
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

print("--- API CONNECTION TEST ---")

# 1. Check Key
key = os.getenv("KALSHI_API_KEY")
if key:
    print(f"✅ Key loaded: {key[:10]}... (Length: {len(key)})")
    if "-----BEGIN RSA PRIVATE KEY-----" in key:
        print("   Format: RSA Key detected (Correct for Kalshi)")
    else:
        print("   Format: Simple String (Might be incorrect if using RSA auth)")
else:
    print("❌ No Key Found in .env")

# 2. Test Public Endpoint
url = "https://api.elections.kalshi.com/trade-api/v2/markets"
print(f"\nTesting Endpoint: {url}")
try:
    response = requests.get(url, params={"limit": 1, "status": "open"})
    print(f"✅ API Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        markets = data.get('markets', [])
        if markets:
            print(f"✅ Successfully fetched {len(markets)} market(s).")
            print(f"   Sample Market: {markets[0].get('ticker')}")
            print(f"   Sample Price (Yes Bid): {markets[0].get('yes_bid')}")
        else:
            print("⚠️ Connected, but no markets returned.")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
