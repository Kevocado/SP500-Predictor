import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Load .env
root_dir = Path(__file__).parent.parent
load_dotenv(dotenv_path=root_dir / '.env', override=True)

class NewsAnalyzer:
    """
    PhD Milestone 1: Bayesian News Aggregator
    Uses Gemini 1.5 Flash to analyze market-moving news and calculate an 'Alpha impact' score.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def get_general_sentiment(self, vix_value, macro_news_snippets):
        """
        Aggregates VIX and News into a single Market Heat score (-100 to +100).
        """
        prompt = f"""
        Analyze current market heat based on VIX and news headlines.
        VIX: {vix_value}
        NEWS: {json.dumps(macro_news_snippets)}
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "heat_score": int (-100 to 100, where 0 is neutral),
            "label": "Bullish" | "Bearish" | "Fearful" | "Greedy",
            "vix_contribution": float,
            "news_contribution": float,
            "summary": "One sentence summary."
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except Exception:
            return {"heat_score": 0, "label": "Neutral", "vix_contribution": 0, "news_contribution": 0, "summary": "Sentiment engine offline."}

    def analyze_event_impact(self, ticker, title, current_prob, news_snippets):
        """
        Analyzes news snippets and returns a probability adjustment.
        
        Args:
            ticker: The Kalshi market ticker.
            title: The market title (e.g., 'Will the Fed raise rates?').
            current_prob: Our model's current probability (0-100).
            news_snippets: A list of recent news headlines/texts.
            
        Returns:
            dict: { 'adjusted_prob': float, 'reasoning': str, 'sentiment': str }
        """
        if not news_snippets:
            return {
                'adjusted_prob': current_prob,
                'reasoning': "No fresh news available for analysis.",
                'sentiment': "Neutral"
            }

        prompt = f"""
        You are an institutional quantitative researcher at a macro hedge fund.
        Analyze the following news snippets for their impact on the specific prediction market:
        
        MARKET: {title} (Ticker: {ticker})
        OUR BASE MODEL PROBABILITY: {current_prob}%
        
        NEWS SNIPPETS:
        {json.dumps(news_snippets, indent=2)}
        
        TASK:
        1. Determine if this news increases or decreases the likelihood of this event occurring.
        2. Provide a 'Bayesian Adjustment' to the probability. 
        3. Be conservative. Adjustments should rarely exceed +/- 15% unless the news is definitive.
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "adjusted_prob": float,
            "impact_score": float (-5 to 5, where 0 is no impact),
            "sentiment": "Bullish" | "Bearish" | "Neutral",
            "reasoning": "One sentence quantitative explanation."
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            # Find JSON block
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            res_json = json.loads(text[start:end])
            return res_json
        except Exception as e:
            return {
                "adjusted_prob": current_prob,
                "impact_score": 0,
                "sentiment": "Error",
                "reasoning": f"Analyzer failed: {str(e)}"
            }

if __name__ == "__main__":
    # Test
    analyzer = NewsAnalyzer()
    test_news = [
        "Fed's Powell says inflation remains stubborn, suggests rates may stay higher for longer.",
        "Consumer price index comes in 0.1% higher than economist expectations."
    ]
    result = analyzer.analyze_event_impact(
        "KXFED-MAR24", 
        "Will the Fed raise target rates in March?", 
        40.0, 
        test_news
    )
    print(json.dumps(result, indent=2))
