"""
Telegram Notifier — Human-in-the-Loop Trade Alerts

Since Kalshi API is read-only, all execution signals are routed here
as formatted, actionable Telegram alerts.

Alert Types:
  - Kill Switch conditions
  - Weather latency arbitrage edges
  - GEX flips / Amihud liquidity cascades
  - VIX emergency (> 45)
  - Model drift warnings

Bot: t.me/KevsWeatherBot
"""

import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip('"').strip("'")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip('"').strip("'")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


class TelegramNotifier:
    """Sends formatted trade alerts to Telegram."""

    def __init__(self):
        self.bot_token = BOT_TOKEN
        self.chat_id = CHAT_ID
        self._resolved_chat_id = None

    def is_enabled(self):
        return bool(self.bot_token and self.chat_id)

    def _get_chat_id(self):
        """
        Resolve the chat ID. If TELEGRAM_CHAT_ID is a username like @Bot,
        we need the numeric ID from getUpdates. Cache after first resolution.
        """
        if self._resolved_chat_id:
            return self._resolved_chat_id

        # If it's already numeric, use directly
        if self.chat_id.lstrip("-").isdigit():
            self._resolved_chat_id = self.chat_id
            return self._resolved_chat_id

        # Try to get numeric chat ID from recent messages to the bot
        try:
            r = requests.get(f"{BASE_URL}/getUpdates", timeout=5)
            if r.status_code == 200:
                updates = r.json().get("result", [])
                for update in updates:
                    msg = update.get("message", {})
                    chat = msg.get("chat", {})
                    if chat.get("id"):
                        self._resolved_chat_id = str(chat["id"])
                        print(f"  📱 Resolved Telegram chat ID: {self._resolved_chat_id}")
                        return self._resolved_chat_id
        except Exception as e:
            print(f"  ⚠️ Failed to resolve Telegram chat ID: {e}")

        # Fallback: use as-is
        self._resolved_chat_id = self.chat_id
        return self._resolved_chat_id

    def send_message(self, text: str, parse_mode: str = "Markdown"):
        """Send a raw text message to the configured Telegram chat."""
        if not self.is_enabled():
            print("  ⚠️ Telegram not configured (missing BOT_TOKEN or CHAT_ID)")
            return False

        chat_id = self._get_chat_id()
        try:
            r = requests.post(
                f"{BASE_URL}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            if r.status_code == 200:
                return True
            else:
                print(f"  ⚠️ Telegram API error {r.status_code}: {r.text}")
                return False
        except Exception as e:
            print(f"  ⚠️ Telegram send failed: {e}")
            return False

    # ── Formatted Alert Methods ─────────────────────────────────────

    def alert_weather_edge(self, city: str, nws_temp: float, action: str, price: float, ticker: str):
        """Weather latency arbitrage alert."""
        text = (
            f"🌤️ *Weather Edge Alert*\n\n"
            f"NWS printed *{nws_temp}°F* for {city}\n"
            f"Suggested: *{action}* at ${price:.2f}\n"
            f"Ticker: `{ticker}`\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self.send_message(text)

    def alert_kill_switch(self, reason: str, positions: list = None):
        """Kill switch condition alert."""
        pos_text = ""
        if positions:
            pos_text = "\n".join([f"  • `{p.get('ticker', '?')}` qty={p.get('total_traded', 0)}" for p in positions[:5]])
            pos_text = f"\n\nOpen Positions:\n{pos_text}"

        text = (
            f"🚨 *KILL SWITCH TRIGGERED*\n\n"
            f"Reason: {reason}\n"
            f"Action: *Liquidate all positions immediately*"
            f"{pos_text}"
        )
        return self.send_message(text)

    def alert_vix_emergency(self, vix_value: float):
        """VIX > 45 emergency liquidation warning."""
        text = (
            f"🔴 *VIX EMERGENCY*\n\n"
            f"VIX has spiked to *{vix_value:.1f}*\n"
            f"Action: *Liquidate all positions immediately*\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self.send_message(text)

    def alert_gex_flip(self, ticker: str, gex_value: float, direction: str):
        """GEX flip alert (positive → negative or vice versa)."""
        emoji = "⚠️" if direction == "negative" else "🟢"
        text = (
            f"{emoji} *Gamma Flip: {ticker}*\n\n"
            f"GEX turned *{direction}* ({gex_value:+.2f})\n"
            f"{'Expect expanded range & increased volatility.' if direction == 'negative' else 'Dealer hedging should stabilize prices.'}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self.send_message(text)

    def alert_liquidity_cascade(self, ticker: str, amihud_ratio: float, sigma_above: float):
        """Amihud illiquidity spike alert."""
        text = (
            f"🚨 *Liquidity Cascade: {ticker}*\n\n"
            f"Amihud ratio: *{amihud_ratio:.6f}* ({sigma_above:.1f}σ above mean)\n"
            f"Action: *Reduce exposure immediately*\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self.send_message(text)

    def alert_model_drift(self, brier_score: float, threshold: float):
        """Model accuracy drift warning."""
        text = (
            f"📉 *Model Drift Detected*\n\n"
            f"Brier Score: *{brier_score:.4f}* (threshold: {threshold:.4f})\n"
            f"Action: Consider retraining or adjusting parameters\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self.send_message(text)

    def alert_scanner_results(self, opportunities: list, min_edge: float = 15.0):
        """Summary of high-edge opportunities from the scanner."""
        high_edge = [o for o in opportunities if float(o.get("edge", 0)) >= min_edge]
        if not high_edge:
            return False

        high_edge.sort(key=lambda x: float(x.get("edge", 0)), reverse=True)
        top = high_edge[:5]

        lines = [f"🤖 *Scanner: {len(high_edge)} opportunities >{min_edge}% edge*\n"]
        for opp in top:
            lines.append(
                f"• *{opp.get('engine', '?')}* | {opp.get('asset', '?')} | "
                f"{opp.get('action', '?')} | Edge: +{opp.get('edge', 0):.1f}%"
            )

        return self.send_message("\n".join(lines))


if __name__ == "__main__":
    print("Testing Telegram Notifier...")
    notifier = TelegramNotifier()

    if not notifier.is_enabled():
        print("  ❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        print("  Tip: Send any message to your bot first, then run this test.")
        exit(1)

    result = notifier.send_message("✅ *SP500 Predictor* — Telegram integration active!")
    if result:
        print("  ✅ Test message sent successfully!")
    else:
        print("  ❌ Failed to send test message.")
        print("  Tip: Send /start to your bot first, then re-run this test.")
