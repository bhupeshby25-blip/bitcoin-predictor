# Bitcoin Predictor Bot

A regime-aware Bitcoin analysis bot that sends **trade signals with conviction scores** to a Telegram channel on a configurable schedule. It leverages machine learning and technical indicators to evaluate market conditions and risk before issuing buy/sell/neutral recommendations.

## Features
- **Regime-Aware Strategy**: Adjusts its signal generation logic based on the current market regime (e.g., High Volatility, Trending, Ranging).
- **Automated Telegram Signals**: Connects directly to a Telegram bot via `python-telegram-bot` to push signals.
- **Dynamic Intervals**: Update analysis intervals on-the-fly using commands like `/setinterval 30m`.
- **Risk Management**: Analyzes and classifies risk per trade, reporting ATR and conviction scores.
- **Predictive Action Matrix**: Robust matrix for exact target bounds.

For detailed setup instructions, including how to configure the Telegram bot and environment variables, please refer to the [Setup Guide](SETUP.md).
