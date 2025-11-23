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

def get_market_status():
    """
    Determines if the US market is open and when the next event (open/close) is.
    
    Returns:
        dict: {
            'is_open': bool,
            'status_text': str,
            'next_event_text': str,
            'color': str
        }
    """
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    is_weekday = now.weekday() < 5
    
    if is_weekday and market_open <= now.time() < market_close:
        # Market is Open
        is_open = True
        status_text = "Market is OPEN"
        color = "green"
        
        # Calculate time to close
        close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
        delta = close_dt - now
        hours, remainder = divmod(delta.seconds, 3600)
        minutes = remainder // 60
        next_event_text = f"Closes in {hours}h {minutes}m"
        
    else:
        # Market is Closed
        is_open = False
        status_text = "Market is CLOSED"
        color = "red"
        
        # Calculate time to next open
        # If before 9:30 AM today
        if is_weekday and now.time() < market_open:
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            day_str = "Today"
        # If after 4 PM Friday or Weekend
        elif now.weekday() >= 4: # Friday(4), Sat(5), Sun(6)
            days_ahead = 7 - now.weekday() # Mon is 0
            next_open = (now + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
            day_str = "Monday"
        # If after 4 PM Mon-Thu
        else:
            next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
            day_str = "Tomorrow"
            
        delta = next_open - now
        hours, remainder = divmod(delta.seconds, 3600)
        # Add days to hours if delta.days > 0 (though usually we just say "Opens Monday")
        total_hours = hours + (delta.days * 24)
        
        next_event_text = f"Opens {day_str} at 9:30 AM ET"

    return {
        'is_open': is_open,
        'status_text': status_text,
        'next_event_text': next_event_text,
        'color': color
    }
