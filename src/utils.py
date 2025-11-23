import pytz
from datetime import datetime, time

def get_market_status(ticker="SPX"):
    """
    Determines if the market is currently open for the given ticker.
    Returns a dict with status details.
    """
    # Crypto is 24/7
    if ticker in ["BTC", "ETH", "BTC-USD", "ETH-USD"]:
        return {
            'is_open': True,
            'is_pre_market': False,
            'status_text': "Market is OPEN (24/7)",
            'next_event_text': "Closes Never",
            'color': "#3b82f6" # Blue for Live
        }

    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    current_time = now.time()
    
    # Market Hours (ET)
    pre_market_open = time(4, 0)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Weekends
    if now.weekday() >= 5: # Sat=5, Sun=6
        return {
            'is_open': False,
            'is_pre_market': False,
            'status_text': "Market is CLOSED (Weekend)",
            'next_event_text': "Opens Monday 9:30 AM ET",
            'color': "#6b7280" # Grey for Closed
        }
        
    # Weekdays
    if pre_market_open <= current_time < market_open:
        # Pre-Market
        return {
            'is_open': False,
            'is_pre_market': True,
            'status_text': "PRE-MARKET",
            'next_event_text': "Opens 9:30 AM ET",
            'color': "#f59e0b" # Orange/Yellow
        }
    elif market_open <= current_time < market_close:
        # Regular Market Hours
        # Calculate time to close
        close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
        delta = close_dt - now
        hours, remainder = divmod(delta.seconds, 3600)
        minutes = remainder // 60
        
        return {
            'is_open': True,
            'is_pre_market': False,
            'status_text': "Market is OPEN",
            'next_event_text': f"Closes in {hours}h {minutes}m",
            'color': "#3b82f6" # Blue for Live
        }
    else:
        # After Hours / Closed
        return {
            'is_open': False,
            'is_pre_market': False,
            'status_text': "Market is CLOSED",
            'next_event_text': "Opens Tomorrow 9:30 AM ET",
            'color': "#6b7280" # Grey for Closed
        }

def determine_best_timeframe(ticker):
    """
    Determines the best timeframe (Hourly vs Daily) based on asset and market status.
    """
    # Crypto -> Always Hourly (Fast paced)
    if ticker in ["BTC", "ETH", "BTC-USD", "ETH-USD"]:
        return "Hourly"
        
    # Stocks -> Hourly if Open, Daily if Closed
    status = get_market_status(ticker)
    if status['is_open']:
        return "Hourly"
    else:
        return "Daily"
