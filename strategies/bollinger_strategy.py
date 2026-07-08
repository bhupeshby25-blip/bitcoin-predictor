"""
Bollinger Band Breakout Strategy
---------------------------------
Best for: HIGH_VOLATILITY / RANGING regimes

Logic:
  - BUY  when price closes above the upper band (upward breakout)
  - SELL when price closes below the lower band (downward breakout)
  - HOLD when price is inside the bands

Prediction: band midline (SMA) projection using linear regression.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class BollingerStrategy(BaseStrategy):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Band Breakout")
        self.period  = period
        self.std_dev = std_dev

    def analyze(self, data: pd.DataFrame) -> dict:
        if len(data) < max(self.period + 5, 55):
            return self._empty(data)

        df = data.copy()
        df['sma']    = df['close'].rolling(self.period).mean()
        df['std']    = df['close'].rolling(self.period).std()
        df['upper']  = df['sma'] + self.std_dev * df['std']
        df['lower']  = df['sma'] - self.std_dev * df['std']
        df['vol_ma'] = df['volume'].rolling(self.period).mean()
        df['ma50']   = df['close'].rolling(50).mean()

        current_price = df['close'].iloc[-1]
        upper    = df['upper'].iloc[-1]
        lower    = df['lower'].iloc[-1]
        sma      = df['sma'].iloc[-1]
        ma50     = df['ma50'].iloc[-1]
        vol_now  = df['volume'].iloc[-1]
        vol_avg  = df['vol_ma'].iloc[-1]
        width_pct = ((upper - lower) / sma) * 100

        prev_price = df['close'].iloc[-2]
        prev_upper = df['upper'].iloc[-2]
        prev_lower = df['lower'].iloc[-2]

        volume_confirmed = vol_now > vol_avg         # volume spike
        trend_up   = current_price > ma50            # above 50MA = bullish bias
        trend_down = current_price < ma50            # below 50MA = bearish bias

        action, reason = "HOLD", "Price inside Bollinger Bands"

        if current_price > upper and prev_price <= prev_upper:
            if volume_confirmed and trend_up:
                action = "BUY"
                reason = f"Bullish breakout above upper band (${upper:,.2f}) — volume & trend confirmed"
            else:
                reason = (
                    f"Breakout above upper (${upper:,.2f}) rejected: "
                    + ("low volume" if not volume_confirmed else "")
                    + (" + bearish trend" if not trend_up else "")
                )

        elif current_price < lower and prev_price >= prev_lower:
            if volume_confirmed and trend_down:
                action = "SELL"
                reason = f"Bearish breakdown below lower band (${lower:,.2f}) — volume & trend confirmed"
            else:
                reason = (
                    f"Breakdown below lower (${lower:,.2f}) rejected: "
                    + ("low volume" if not volume_confirmed else "")
                    + (" + bullish trend" if not trend_down else "")
                )

        predicted_price, predicted_chg = self._predict_next(df['sma'])

        return {
            "action":               action,
            "price":                current_price,
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_chg,
            "reason":               reason,
            "metadata": {
                "Upper Band":  round(upper, 2),
                "SMA20":       round(sma, 2),
                "MA50":        round(ma50, 2),
                "Lower Band":  round(lower, 2),
                "Band Width":  f"{width_pct:.1f}%",
                "Vol Confirm": "✅" if volume_confirmed else "❌"
            }
        }


    def _predict_next(self, sma_series: pd.Series, lookback: int = 20):
        try:
            y = sma_series.dropna().tail(lookback).values
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
