import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# NOAA Grid Endpoints (found via coordinates)
# Central Park, NY: 40.7826, -73.9656 -> OKX/33,35
# O'Hare, Chicago: 41.9742, -87.9073 -> LOT/73,73

LOCATIONS = {
    "NYC": {
        "name": "New York",
        "grid_url": "https://api.weather.gov/gridpoints/OKX/33,35/forecast",
        "kalshi_prefix": "KXNYC" # e.g. KXNYC-24DEC15-T50
    },
    "CHI": {
        "name": "Chicago",
        "grid_url": "https://api.weather.gov/gridpoints/LOT/73,73/forecast",
        "kalshi_prefix": "KXCHI"
    }
}

def fetch_noaa_forecast(grid_url):
    """
    Fetches the 7-day forecast from NOAA API and extracts daily high temperatures.
    """
    headers = {
        "User-Agent": "(SP500_Predictor, contact@example.com)"
    }

    try:
        response = requests.get(grid_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"NOAA API Error: {response.status_code}")
            return None

        data = response.json()
        periods = data.get("properties", {}).get("periods", [])

        forecasts = {}
        for period in periods:
            # We only care about daytime highs for these typical Kalshi markets
            if period.get("isDaytime"):
                date_str = period.get("startTime")[:10] # YYYY-MM-DD
                temp = period.get("temperature")
                forecasts[date_str] = temp

        return forecasts
    except Exception as e:
        print(f"Error fetching NOAA data: {e}")
        return None

def fetch_kalshi_market_data(ticker):
    """
    Fetches market pricing from Kalshi for a given ticker.
    """
    api_key = os.getenv("KALSHI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        r = requests.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook",
            headers=headers, timeout=5
        )
        if r.status_code != 200:
            return None

        data = r.json().get('orderbook', {})
        yes_bids = data.get('yes', []) or []
        no_bids = data.get('no', []) or []

        best_yes = yes_bids[-1][0] if yes_bids else 0
        best_no = no_bids[-1][0] if no_bids else 0

        if best_yes == 0 and best_no == 0:
            return None

        # Implied probability is roughly the yes price / 100
        implied_prob = best_yes / 100.0 if best_yes > 0 else (100 - best_no) / 100.0

        return {
            "best_yes_cents": best_yes,
            "best_no_cents": best_no,
            "implied_prob": implied_prob
        }
    except Exception as e:
        return None

def calculate_weather_edge():
    """
    Compares NOAA forecast data against Kalshi daily high markets to find > 5% edge.
    """
    opportunities = []

    for loc_code, loc_info in LOCATIONS.items():
        print(f"Fetching NOAA forecast for {loc_info['name']}...")
        forecasts = fetch_noaa_forecast(loc_info['grid_url'])

        if not forecasts:
            continue

        # For this engine, we'll look at tomorrow's forecast to compare against markets
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        if tomorrow not in forecasts:
            print(f"Could not find tomorrow's forecast for {loc_info['name']}.")
            continue

        predicted_high = forecasts[tomorrow]
        print(f"  Tomorrow's Predicted High for {loc_code}: {predicted_high}°F")

        # In a real scenario, you'd query the Kalshi API for active markets on this date.
        # We will simulate checking a specific strike price close to the predicted high.
        # Let's say we check if the temperature will be greater than predicted_high - 2

        test_strike = predicted_high - 2
        date_formatted = (datetime.now() + timedelta(days=1)).strftime('%y%b%d').upper() # e.g. 24DEC15
        simulated_ticker = f"{loc_info['kalshi_prefix']}-{date_formatted}-T{test_strike}"

        # Our internal model: The NOAA forecast is highly accurate 1 day out.
        # If predicted high > strike + 2 degrees, we assign a 90% probability.
        our_prob = 0.90 if predicted_high > test_strike else 0.10

        market_data = fetch_kalshi_market_data(simulated_ticker)

        if market_data:
            market_prob = market_data["implied_prob"]
            edge = our_prob - market_prob

            if abs(edge) > 0.05:
                opportunities.append({
                    "PartitionKey": "WEATHER_ENGINE",
                    "RowKey": f"{loc_code}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "MarketTicker": simulated_ticker,
                    "SignalType": f"Daily High {loc_code}",
                    "CalculatedProb": our_prob,
                    "MarketProb": market_prob,
                    "Edge": round(edge, 4),
                    "Action": "BUY YES" if edge > 0 else "BUY NO",
                    "Reasoning": f"NOAA forecasts {predicted_high}°F for {tomorrow}. Our prob for >{test_strike}°F is {our_prob:.2f} but market is at {market_prob:.2f}",
                    "Timestamp": datetime.utcnow().isoformat()
                })

    return opportunities

if __name__ == "__main__":
    print("Running Weather Engine...")
    ops = calculate_weather_edge()
    if not ops:
        print("No weather opportunities with > 5% edge found (or unable to fetch market data).")
    for op in ops:
        print(op)
