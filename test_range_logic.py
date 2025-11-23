#!/usr/bin/env python3
"""Test script for Range Logic verification."""

import pandas as pd
import numpy as np
from scipy.stats import norm
from zoneinfo import ZoneInfo
from datetime import datetime

# Mock calculate_probability
def calculate_probability(prediction, strike, rmse):
    z_score = (prediction - strike) / rmse
    prob = norm.cdf(z_score) * 100
    return prob

# Mock Data
curr_price = 3400
pred = 3420
rmse = 50

# Mock Range Market
m = {
    'ticker': 'ETH',
    'floor_strike': 3400,
    'cap_strike': 3450,
    'yes_ask': 40,
    'expiration': '2025-11-30T22:00:00Z'
}

print(f"Current Price: {curr_price}")
print(f"Prediction: {pred}")
print(f"Range: {m['floor_strike']} - {m['cap_strike']}")
print(f"Cost (Ask): {m['yes_ask']}")

# Logic Test
lower = m['floor_strike']
upper = m['cap_strike']

# 1. Proximity
mid_point = (lower + upper) / 2
pct_diff = abs(mid_point - curr_price) / curr_price
print(f"Proximity: {pct_diff:.4f} (Threshold: 0.02)")

if pct_diff <= 0.02:
    print("✅ Proximity Check Passed")
else:
    print("❌ Proximity Check Failed")

# 2. Probability
prob_lower = calculate_probability(pred, lower, rmse)
prob_upper = calculate_probability(pred, upper, rmse)
prob_in_range = prob_lower - prob_upper
prob_in_range = max(0.0, min(100.0, prob_in_range))

print(f"Prob > {lower}: {prob_lower:.1f}%")
print(f"Prob > {upper}: {prob_upper:.1f}%")
print(f"Prob In Range: {prob_in_range:.1f}%")

# 3. Edge
cost = m['yes_ask']
edge = prob_in_range - cost
print(f"Edge: {edge:.1f}% (Threshold: 5%)")

if edge > 5:
    print("✅ Edge Check Passed")
else:
    print("❌ Edge Check Failed")
