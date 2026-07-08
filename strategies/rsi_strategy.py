"""
RSI Mean Reversion Strategy
---------------------------------
Best for: RANGING / BEAR_TREND regimes

Logic:
  - BUY  when RSI drops below oversold threshold (price likely undervalued)
  - SELL when RSI climbs above overbought threshold (price likely overvalued)
  - HOLD otherwise

Prediction: short-term momentum from RSI velocity.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Mean Reversion")
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought

    def analyze(self, data: pd.DataFrame) -> dict:
        if len(data) < self.period + 5:
            return self._empty(data)

        rsi = self._calculate_rsi(data['close'])
        current_rsi   = rsi.iloc[-1]
        prev_rsi      = rsi.iloc[-2]
        current_price = data['close'].iloc[-1]

        action, reason = "HOLD", "RSI in neutral zone"

        if current_rsi < self.oversold and prev_rsi >= self.oversold:
            action = "BUY"
            reason = f"RSI entered oversold zone ({current_rsi:.1f} < {self.oversold})"
        elif current_rsi > self.overbought and prev_rsi <= self.overbought:
            action = "SELL"
            reason = f"RSI entered overbought zone ({current_rsi:.1f} > {self.overbought})"
        elif current_rsi < self.oversold:
            action = "BUY"
            reason = f"RSI remains oversold ({current_rsi:.1f})"
        elif current_rsi > self.overbought:
            action = "SELL"
            reason = f"RSI remains overbought ({current_rsi:.1f})"

        predicted_price, predicted_chg = self._predict_next(data['close'])

        return {
            "action":               action,
            "price":                current_price,
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_chg,
            "reason":               reason,
            "metadata":             {"RSI": round(current_rsi, 1), "Oversold": self.oversold, "Overbought": self.overbought}
        }

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/self.period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/self.period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

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
