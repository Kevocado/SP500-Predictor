import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY missing from environment.")
        return None

    genai.configure(api_key=api_key)
    # Using the gemini-2.5-flash model for fast, cheap reasoning, or pro if preferred
    return genai.GenerativeModel('gemini-2.5-flash')

def scrutinize_trade(trade_opportunity):
    """
    Sends a mathematically identified trade to Gemini to act as a qualitative filter.

    Args:
        trade_opportunity (dict): Dictionary containing details of the trade.

    Returns:
        tuple: (bool is_approved, str reasoning)
    """
    model = get_gemini_client()
    if not model:
        return False, "Gemini API key not configured."

    signal_type = trade_opportunity.get('SignalType', 'Unknown')
    market_ticker = trade_opportunity.get('MarketTicker', 'Unknown')
    action = trade_opportunity.get('Action', 'Unknown')
    reasoning = trade_opportunity.get('Reasoning', '')

    prompt = f"""
You are an expert financial and geopolitical risk analyst.
Our quantitative engine has identified a statistical edge in a prediction market.

Trade Details:
- Market: {market_ticker}
- Signal Type: {signal_type}
- Recommended Action: {action}
- Quantitative Reasoning: {reasoning}

Your task is to act as a "Scrutinizer". Is this mathematical edge real, or is the market pricing in a qualitative event that the historical math missed?
For example:
- If this is a weather market, is there a sudden unseasonable polar vortex or hurricane that the 7-day trailing models missed?
- If this is a macroeconomic market, did the Fed just make an unscheduled announcement? Is there breaking news that makes the historical data irrelevant?
- Is this a "value trap"?

Respond in the following format exactly:
APPROVED: [TRUE or FALSE]
REASON: [1-2 sentences explaining why the trade is approved or rejected based on qualitative factors]
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse the response
        is_approved = False
        parsed_reasoning = "Could not parse Gemini response."

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('APPROVED:'):
                val = line.replace('APPROVED:', '').strip().upper()
                if 'TRUE' in val:
                    is_approved = True
            elif line.startswith('REASON:'):
                parsed_reasoning = line.replace('REASON:', '').strip()

        return is_approved, parsed_reasoning

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return False, f"API Error: {e}"

if __name__ == "__main__":
    # Test the scrutinizer
    test_trade = {
        "MarketTicker": "KXCPI-24DEC-T0.2",
        "SignalType": "Core CPI",
        "Action": "BUY YES",
        "Reasoning": "Calculated MoM CPI is 0.3%, our prob is 0.90 but market prices at 0.50"
    }

    print("Testing Gemini Scrutinizer...")
    approved, reason = scrutinize_trade(test_trade)
    print(f"Approved: {approved}")
    print(f"Reason: {reason}")
