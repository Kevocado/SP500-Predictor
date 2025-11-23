import numpy as np

def generate_trading_signals(ticker, predicted_price, current_price, rmse):
    """
    Generates trading signals for both Direction (Strikes) and Volatility (Ranges).
    
    Args:
        ticker (str): Asset symbol.
        predicted_price (float): The model's predicted price.
        current_price (float): The current market price.
        rmse (float): Model uncertainty/RMSE.
        
    Returns:
        dict: {
            'strikes': list of dicts (Strike opportunities),
            'ranges': list of dicts (Range bucket opportunities)
        }
    """
    signals = {
        'strikes': [],
        'ranges': []
    }
    
    # --- 1. Strike Signals (Direction) ---
    # Generate strikes around the current price
    # Step size depends on asset price
    if current_price > 10000: # BTC-like
        step = 100
        buffer = 50
    elif current_price > 1000: # ETH/SPX-like
        step = 10
        buffer = 5
    else:
        step = 1
        buffer = 0.5
        
    base_price = round(current_price / step) * step
    # Check strikes in a wider window
    strikes_to_check = [base_price + (k * step) for k in range(-5, 6)]
    
    for strike in strikes_to_check:
        # Simple Directional Logic (as requested)
        # "If predicted_price > strike_price + buffer, signal is BUY YES"
        # We can also use the probabilistic approach for the "Prob" column
        
        # Calculate Probability (using Z-score approach from model.py logic)
        # Z = (Predicted - Strike) / RMSE
        # Prob = CDF(Z)
        # We'll re-implement a simple version here or just use the logic requested.
        # Let's use the logic requested for the "Action" but calculate Prob for display.
        
        import scipy.stats as stats
        z_score = (predicted_price - strike) / rmse
        prob = stats.norm.cdf(z_score) * 100
        
        action = "PASS"
        # Logic from prompt:
        # If predicted_price > strike + buffer -> Bullish (Buy YES on > Strike)
        # If predicted_price < strike - buffer -> Bearish (Buy NO on > Strike)
        
        # Note: "Buy NO on > Strike" is equivalent to "Betting Price < Strike"
        
        if predicted_price > strike + buffer:
            action = "🟢 BUY YES" # Expect Price > Strike
        elif predicted_price < strike - buffer:
            action = "🔴 BUY NO"  # Expect Price < Strike
            
        # Filter for interesting actions only? Or return all?
        # Let's return all but highlight interesting ones
        if action != "PASS":
            signals['strikes'].append({
                "Strike": f"> ${strike}",
                "Prob": f"{prob:.1f}%",
                "Action": action,
                "Raw_Strike": strike # For sorting if needed
            })

    # --- 2. Range Signals (Volatility) ---
    # Define range buckets
    # For BTC: 1000 point increments? Prompt says "1000 point increments for BTC, 50-100 for ETH"
    # Let's make it dynamic based on price
    
    if ticker in ["BTC", "BTC-USD"]:
        range_step = 500 # 1000 might be too wide if volatility is low, let's try 500 or 1000. Prompt said 1000.
        range_step = 1000
    elif ticker in ["ETH", "ETH-USD"]:
        range_step = 100
    elif ticker in ["SPX", "NDX", "Nasdaq"]:
        range_step = 50 # SPX usually moves 20-50 points
    else:
        range_step = 10
        
    # Generate ranges covering the predicted price
    # Center around predicted price
    pred_base = (predicted_price // range_step) * range_step
    
    # Create a few ranges around the prediction
    ranges_to_check = []
    for k in range(-2, 3):
        start = pred_base + (k * range_step)
        end = start + range_step
        ranges_to_check.append((start, end))
        
    for r_start, r_end in ranges_to_check:
        # Check if predicted price falls in this range
        in_range = r_start <= predicted_price < r_end
        
        action = "PASS"
        if in_range:
            action = "🟢 BUY YES"
            
        signals['ranges'].append({
            "Range": f"${r_start:,.0f} - ${r_end:,.0f}",
            "Predicted In Range?": "✅ YES" if in_range else "❌ NO",
            "Action": action,
            "Is_Winner": in_range
        })
        
    return signals
