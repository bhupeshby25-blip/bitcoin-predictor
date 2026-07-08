"""
MACD Momentum Strategy
---------------------------------
Best for: BULL_TREND regime

Logic:
  - BUY  when MACD line crosses above Signal line (bullish momentum)
  - SELL when MACD line crosses below Signal line (bearish momentum)
  - HOLD otherwise

Prediction: MACD histogram trend projected one step forward.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD Momentum")
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def analyze(self, data: pd.DataFrame) -> dict:
        if len(data) < self.slow + self.signal + 5:
            return self._empty(data)

        closes = data['close']
        ema_fast   = closes.ewm(span=self.fast,   adjust=False).mean()
        ema_slow   = closes.ewm(span=self.slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram  = macd_line - signal_line

        macd_curr   = macd_line.iloc[-1]
        macd_prev   = macd_line.iloc[-2]
        sig_curr    = signal_line.iloc[-1]
        sig_prev    = signal_line.iloc[-2]
        hist_curr   = histogram.iloc[-1]
        current_price = closes.iloc[-1]

        action, reason = "HOLD", "No MACD crossover"

        # Bullish crossover
        if macd_prev <= sig_prev and macd_curr > sig_curr:
            action = "BUY"
            reason = f"MACD bullish crossover (histogram: {hist_curr:+.2f})"
        # Bearish crossover
        elif macd_prev >= sig_prev and macd_curr < sig_curr:
            action = "SELL"
            reason = f"MACD bearish crossover (histogram: {hist_curr:+.2f})"

        predicted_price, predicted_chg = self._predict_next(closes)

        return {
            "action":               action,
            "price":                current_price,
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_chg,
            "reason":               reason,
            "metadata": {
                "MACD":      round(macd_curr, 2),
                "Signal":    round(sig_curr,  2),
                "Histogram": round(hist_curr, 2)
            }
        }

    def _predict_next(self, prices: pd.Series, lookback: int = 20):
        try:
            y = prices.dropna().tail(lookback).values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * len(y) + intercept
            change_pct = ((predicted - y[-1]) / y[-1]) * 100
            return round(predicted, 2), round(change_pct, 2)
        except Exception:
            return None, None

    def _empty(self, data):
        return {"action": "HOLD", "price": data['close'].iloc[-1],
                "predicted_price": None, "predicted_change_pct": None,
                "reason": "Insufficient data", "metadata": {}}
