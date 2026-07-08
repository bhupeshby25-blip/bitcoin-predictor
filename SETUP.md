# Bitcoin Predictor Bot — Setup Guide

A regime-aware Bitcoin analysis bot that sends **trade signals with conviction scores** to a Telegram channel on a configurable schedule.

---

## Prerequisites

- Python 3.10 or higher
- A Telegram account
- Internet connection (fetches live BTC data from Binance via `ccxt` and Fear & Greed Index from `alternative.me`)

---

## Step 1 — Clone / Open the Project

```bash
cd /Users/bhupesh/Documents/bitcoin-predictor
```

---

## Step 2 — Create a Virtual Environment

```bash
python3 -m venv env
source env/bin/activate        # macOS / Linux
# env\Scripts\activate         # Windows
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs:

| Package | Purpose |
|---|---|
| `pandas` / `numpy` | Data processing |
| `scikit-learn` / `lightgbm` | ML models |
| `ccxt` | Fetches live OHLCV data from Binance |
| `python-telegram-bot` | Sends signals to Telegram |
| `python-dotenv` | Loads secrets from `.env` |

> **Note:** If `scikit-learn` or `lightgbm` are missing from `requirements.txt`, install them manually:
> ```bash
> pip install scikit-learn lightgbm
> ```

---

## Step 4 — Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts to name your bot
3. Copy the **Bot Token** BotFather gives you (looks like `123456789:AAH...`)

**Get your Chat ID:**
1. Add your new bot to a Telegram group or channel
2. Make it an **Admin** in that group
3. Send any message in the group, then open this URL in a browser:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
4. Find the `"chat"` → `"id"` value in the JSON response. For groups/channels it starts with `-100...`

---

## Step 5 — Configure the `.env` File

Create or edit the `.env` file in the project root:

```bash
# .env
TELEGRAM_BOT_TOKEN=123456789:AAHyour_token_here
TELEGRAM_CHAT_ID=-1001234567890
```

> ⚠️ **Never commit this file to Git.** It's already in `.gitignore`.

---

## Step 6 — (Optional) Adjust Settings

Edit `config.py` to change trading settings:

```python
SYMBOL              = "BTC/USDT"   # Trading pair
TIMEFRAME           = "1h"         # Candle size: 1h, 4h, 1d
LIMIT               = 100          # Number of candles to fetch
DEFAULT_RISK_PER_TRADE = 0.02      # 2% capital risked per trade
```

---

## Step 7 — Run the Bot

```bash
source env/bin/activate
python main.py
```

The bot will start and immediately send its first analysis to your Telegram channel. After that, it runs every **1 hour** by default.

---

## Telegram Commands

Once the bot is running, you can control it from Telegram:

| Command | Description |
|---|---|
| `/start` | Start the bot and trigger the first analysis |
| `/status` | Show current regime, last signal, and interval |
| `/setinterval 30m` | Change the analysis interval (e.g. `30m`, `4h`, `12h`) |
| `/help` | List all commands |

---

## What the Signal Looks Like

```
⚫ NEUTRAL — BTC/USDT  |  Confidence: 50%

💰 Price Now: $68,320.50
🌍 Market Regime: HIGH_VOLATILITY 🌪️

⚡ Action Signal  : NEUTRAL
🎯 Conviction     : 50%
📊 Volatility Base: 2.64%
💬 Analysis       : Prediction magnitude falls within normal background noise. No statistically significant edge.

🛡 Risk: 🟡 Medium  |  ATR: 2.1%
```

**Signal types:**
- `STRONG_BUY` / `STRONG_SELL` — High-conviction entry, predicted move is **1.5× above** normal volatility
- `WEAK_BUY` / `WEAK_SELL` — Moderate edge, predicted move is **1.0–1.5×** normal volatility
- `NEUTRAL` — No significant edge detected, stay out

---

## Running as a Background Service (macOS)

To keep the bot alive after closing the terminal, use a `launchd` plist or simply run with `nohup`:

```bash
nohup python main.py > bot.log 2>&1 &
echo "Bot running with PID $!"
```

To stop it:
```bash
kill $(pgrep -f main.py)
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `TELEGRAM_BOT_TOKEN not set` | Check your `.env` file exists and the variable name is correct |
| `chat not found` | Make sure the bot is an Admin in the group/channel |
| `No data received` | Check internet connection; Binance may be blocked in your region |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the virtual env |
