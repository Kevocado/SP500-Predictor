"""
AI Scrutinizer - Two-Tier Validation (HuggingFace + Gemini)

PURPOSE: Prevent "value traps" where mathematical edge exists but qualitative factors invalidate it.

TIER 1: Hugging Face (free, local) - handles ~70% of validations
TIER 2: Gemini 1.5 Flash (API) - handles uncertain/complex cases

EXAMPLES OF VALUE TRAPS:
- Model says BTC going to $100k, but SEC just announced Binance investigation
- Weather model says 85°F, but sudden cold front moving in (not in historical data)
- CPI edge exists, but Fed Chair just gave surprise dovish speech

The AI reads breaking news and context to validate that the mathematical edge is REAL.
"""

import google.generativeai as genai
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class AIValidator:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        # Add Hugging Face pre-filter
        try:
            from src.sentiment_filter import SentimentFilter
            self.sentiment_filter = SentimentFilter()
        except Exception as e:
            print(f"⚠️ HuggingFace SentimentFilter not available: {e}")
            self.sentiment_filter = None

        # Track API usage
        self.gemini_calls = 0
        self.hf_auto_approved = 0

    def validate_trade(self, opportunity, recent_news=None):
        """
        Validate a trade using two-tier system:
        1. First: Hugging Face sentiment filter (free, local)
        2. Only if needed: Gemini API (paid)

        This saves ~70% of Gemini API costs.

        Returns:
            dict: {
                'approved': bool,
                'ai_reasoning': str,
                'risk_factors': list,
                'confidence': int (1-10),
                'tier': str ('huggingface'|'gemini'|'fallback'),
                'ai_used': bool,
                'error': str or None
            }
        """

        # TIER 1: Hugging Face Pre-Filter (only for Macro trades)
        if self.sentiment_filter and opportunity.get('engine') == 'Macro':
            try:
                pre_filter = self.sentiment_filter.pre_filter_macro_trade(
                    opportunity,
                    recent_news=recent_news
                )

                if pre_filter['auto_approve']:
                    self.hf_auto_approved += 1
                    return {
                        'approved': True,
                        'ai_reasoning': f"[HuggingFace] {pre_filter['reasoning']}",
                        'risk_factors': [],
                        'confidence': 8,
                        'tier': 'huggingface',
                        'ai_used': True,
                        'error': None
                    }

                elif pre_filter['auto_reject']:
                    return {
                        'approved': False,
                        'ai_reasoning': f"[HuggingFace] {pre_filter['reasoning']}",
                        'risk_factors': ['Sentiment conflict detected'],
                        'confidence': 2,
                        'tier': 'huggingface',
                        'ai_used': True,
                        'error': None
                    }

                # If not auto-approved/rejected, fall through to Gemini
            except Exception as e:
                print(f"HuggingFace pre-filter error: {e}")

        # TIER 2: Gemini API (for uncertain cases or non-Macro trades)
        self.gemini_calls += 1

        prompt = f"""
You are a professional risk analyst for a quantitative trading desk. Your job is to identify "value traps" - trades that look mathematically profitable but have hidden qualitative risks.

PROPOSED TRADE:
- Engine: {opportunity.get('engine', 'Unknown')}
- Asset: {opportunity.get('asset', 'Unknown')}
- Market: {opportunity.get('market_title', 'Unknown')}
- Action: {opportunity.get('action', 'Unknown')}
- Mathematical Edge: {opportunity.get('edge', 0):.1f}%
- Model Reasoning: {opportunity.get('reasoning', 'N/A')}
- Data Source: {opportunity.get('data_source', 'N/A')}

CONTEXT:
Current Date: {datetime.now().strftime('%Y-%m-%d')}

TASK:
1. Identify if there are any breaking news events, weather anomalies, or economic surprises that would invalidate this mathematical edge
2. Check if the model might be missing non-quantitative factors (e.g., political events, natural disasters, policy changes)
3. Assess if the data source is truly predictive or just correlative

Return your analysis in this exact JSON format:
{{
    "approved": true/false,
    "confidence": 1-10,
    "reasoning": "Brief explanation of your decision",
    "risk_factors": ["factor1", "factor2"]
}}

RULES:
- If you don't have recent news/context, approve the trade (trust the math)
- Only reject if you have SPECIFIC concerns (not general skepticism)
- Weather trades with NWS data: almost always approve (NWS is settlement source)
- Macro trades with FRED data: approve unless Fed surprise announcement
- Be concise (max 100 words)
"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text

            # Parse JSON response
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]

            result = json.loads(text.strip())

            return {
                'approved': result.get('approved', False),
                'ai_reasoning': f"[Gemini] {result.get('reasoning', 'No reasoning provided')}",
                'risk_factors': result.get('risk_factors', []),
                'confidence': result.get('confidence', 5),
                'tier': 'gemini',
                'ai_used': True,
                'error': None
            }

        except Exception as e:
            error_msg = str(e)
            print(f"AI Validator error: {error_msg}")
            # On error, default to APPROVING (trust the math) but NOTIFY that AI was not used
            return {
                'approved': True,
                'ai_reasoning': f"⚠️ AI VALIDATION FAILED — trade approved on math only. Error: {error_msg}",
                'risk_factors': ['AI validation unavailable'],
                'confidence': 5,
                'tier': 'fallback',
                'ai_used': False,
                'error': error_msg
            }

    def get_stats(self):
        """Return usage statistics"""
        total_validations = self.gemini_calls + self.hf_auto_approved
        savings = (self.hf_auto_approved / total_validations * 100) if total_validations > 0 else 0

        return {
            'total_validations': total_validations,
            'gemini_api_calls': self.gemini_calls,
            'huggingface_auto_approved': self.hf_auto_approved,
            'api_cost_savings': f"{savings:.1f}%"
        }


if __name__ == "__main__":
    print("Testing AI Validator...")
    try:
        v = AIValidator()
        test_trade = {
            'engine': 'Weather',
            'asset': 'NYC',
            'market_title': 'NYC daily high temperature above 85°F',
            'action': 'BUY YES',
            'edge': 15.0,
            'reasoning': 'NWS forecasts 88°F with 90% confidence. Market underpriced.',
            'data_source': 'NWS Official API (Settlement Source)'
        }
        result = v.validate_trade(test_trade)
        print(f"  Approved: {result['approved']}")
        print(f"  Reasoning: {result['ai_reasoning']}")
        print(f"  Tier: {result['tier']}")
        print(f"  AI Used: {result['ai_used']}")
        if result['error']:
            print(f"  Error: {result['error']}")
    except ValueError as e:
        print(f"  {e}")
