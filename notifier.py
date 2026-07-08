import asyncio
from telegram import Bot
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TelegramNotifier:
    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        
    async def _send(self, message):
        """Internal async sender."""
        bot = Bot(token=self.token)
        await bot.send_message(chat_id=self.chat_id, text=message)

    def send_message(self, message):
        """
        Send a message to the configured Telegram channel.
        Safe to call from synchronous code.
        """
        if not self.token or not self.chat_id:
            print(f"[SIMULATION] Telegram Notification: {message}")
            return

        try:
            asyncio.run(self._send(message))
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
