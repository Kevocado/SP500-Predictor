import logging
import sys
from datetime import datetime, time, timedelta
import pytz

def setup_logger(name=__name__):
    """Sets up a simple logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def get_market_status(ticker="SPX"):
    """
    Determines if the market is open for a specific asset.
    
    Args:
        ticker (str): The asset symbol (e.g., "SPX", "BTC").
        
    Returns:
        dict: {
            'is_open': bool,
            'status_text': str,
            'next_event_text': str,
            'color': str
        }
    """
    # Crypto is ALWAYS Open
    if ticker in ["BTC", "ETH", "BTC-USD", "ETH-USD"]:
        return {
            'is_open': True,
            'is_pre_market': False,
            'status_text': "Market is OPEN (24/7)",
            'next_event_text': "Closes Never",
            'color': "green"
        }

    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    pre_market_open = time(4, 0)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    is_weekday = now.weekday() < 5
    current_time = now.time()
    
    status_text = "Market is CLOSED"
    color = "red"
    is_open = False
    is_pre_market = False
    
    if is_weekday:
        if market_open <= current_time < market_close:
            # Regular Market Hours
            is_open = True
            status_text = "Market is OPEN"
            color = "green"
            
            # Calculate time to close
            close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
            delta = close_dt - now
            hours, remainder = divmod(delta.seconds, 3600)
            minutes = remainder // 60
            next_event_text = f"Closes in {hours}h {minutes}m"
            
        elif pre_market_open <= current_time < market_open:
            # Pre-Market
            is_open = True # Treat as open for prediction purposes
            is_pre_market = True
            status_text = "Pre-Market OPEN"
            color = "orange"
            
            # Calculate time to open
            open_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
            delta = open_dt - now
            hours, remainder = divmod(delta.seconds, 3600)
            minutes = remainder // 60
            next_event_text = f"Opens in {hours}h {minutes}m"
            
        else:
            # Closed (After hours or early morning)
            is_open = False
            
            if current_time >= market_close:
                # After Close
                next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
                day_str = "Tomorrow"
                if now.weekday() == 4: # Friday
                    next_open = (now + timedelta(days=3)).replace(hour=9, minute=30, second=0, microsecond=0)
                    day_str = "Monday"
            else:
                # Early Morning before 4am
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                day_str = "Today"
                
            delta = next_open - now
            hours, remainder = divmod(delta.seconds, 3600)
            total_hours = hours + (delta.days * 24)
            next_event_text = f"Opens {day_str} at 9:30 AM ET"
            
    else:
        # Weekend
        is_open = False
        days_ahead = 7 - now.weekday() # Mon is 0
        next_open = (now + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
        delta = next_open - now
        next_event_text = "Opens Monday at 9:30 AM ET"

    return {
        'is_open': is_open,
        'is_pre_market': is_pre_market,
        'status_text': status_text,
        'next_event_text': next_event_text,
        'color': color
    }
