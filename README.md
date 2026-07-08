<div align="center">
  <h1>📈 Bitcoin Predictor & Trading Bot</h1>
  <p>A sophisticated, regime-aware crypto analysis bot that delivers high-conviction trade signals directly to your Telegram.</p>
</div>

---

## 🌟 Overview

The **Bitcoin Predictor Bot** is an automated analysis engine that monitors live Bitcoin data, evaluates the current market regime (volatility, trend, range), and applies machine learning and technical analysis to generate actionable trading signals. 

Unlike basic bots that just cross moving averages, this system calculates a **conviction score** based on market context and dynamically sizes positions using Advanced Risk Management (ATR-based stop losses and dynamic risk allocation).

## ✨ Core Features

- **🧠 Regime-Aware Intelligence:** Automatically detects whether the market is trending, ranging, or highly volatile, and adjusts strategy parameters accordingly.
- **📊 Predictive Action Matrix:** Replaces random direction flips with exact target bounds. It calculates conviction probabilities (e.g., `STRONG_BUY`, `WEAK_SELL`, `NEUTRAL`).
- **🛡️ Institutional Risk Management:** 
  - Dynamic position sizing based on account risk (default 2%).
  - Average True Range (ATR) based Stop Losses.
- **📱 Telegram Integration:** Complete control via Telegram. Receive automated, richly formatted signals and manage intervals on-the-fly (`/setinterval`).
- **⏱️ Configurable Cron Engine:** Powered by `python-telegram-bot`'s JobQueue for robust, non-blocking asynchronous task scheduling.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Market Data:** [ccxt](https://github.com/ccxt/ccxt) (Binance live OHLCV data) + Fear & Greed Index API
- **Machine Learning & Math:** `scikit-learn`, `lightgbm`, `pandas`, `numpy`
- **Bot Framework:** `python-telegram-bot` (v20+)

---

## ⚙️ Quick Setup

### 1. Installation
```bash
git clone https://github.com/yourusername/bitcoin-predictor.git
cd bitcoin-predictor

# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Telegram Bot Configuration
1. Message **@BotFather** on Telegram with `/newbot`.
2. Get your **Bot Token**.
3. Add the bot to a Channel or Group and make it an Admin.
4. Get your **Chat ID** (usually starts with `-100`).

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=-100your_chat_id_here
```

### 4. Run the Bot
```bash
python main.py
```
*The bot will immediately send its first analysis and then run continuously based on the default interval.*

---

## 🕹️ Telegram Commands

Control the bot directly from your chat:

| Command | Description |
|---|---|
| `/start` | Show bot status and welcome message. |
| `/status` | View the most recent signal, market regime, and risk metrics. |
| `/setinterval <time>` | Change how often signals are generated (e.g., `/setinterval 1h`, `/setinterval 30m`). |
| `/help` | List all available commands. |

## 📊 Signal Breakdown

When the bot fires, you receive a detailed breakdown:
- **Action & Conviction:** What to do and how confident the model is (e.g., `STRONG BUY (85%)`).
- **Price Target:** Predicted magnitude of the move.
- **Market Regime:** Explains *why* the bot chose its strategy (e.g., `HIGH_VOLATILITY 🌪️`).
- **Risk Metrics:** Your exact Stop Loss level and recommended position size (in BTC) based on a 2% capital risk rule.

---

## 🤝 Contributing
Found a bug or have a profitable new strategy to add? Pull requests are welcome! Please ensure you test any strategy additions in the `backtester.py` sandbox before submitting.

## 📄 License
Open-source under the MIT License. Trade at your own risk!
