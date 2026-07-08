import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Trading Settings
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 100  # Number of candles to fetch

# Risk Management Defaults
DEFAULT_RISK_PER_TRADE = 0.02  # 2% of capital
MAX_OPEN_POSITIONS = 1
