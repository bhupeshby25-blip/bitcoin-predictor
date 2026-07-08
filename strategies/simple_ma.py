from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class SimpleMAStrategy(BaseStrategy):
    def __init__(self, short_window=10, long_window=50):
        super().__init__("Simple Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window

    def analyze(self, data: pd.DataFrame) -> dict:
        """
        Detect Golden/Death Cross and predict next candle price via linear regression.
        """
        if len(data) < self.long_window:
            return {
                "action": "HOLD",
                "price": data['close'].iloc[-1],
                "predicted_price": None,
                "predicted_change_pct": None,
                "reason": "Insufficient data for MA calculation",
                "metadata": {}
            }

        df = data.copy()
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma']  = df['close'].rolling(window=self.long_window).mean()

        current_price   = df['close'].iloc[-1]
        short_ma_curr   = df['short_ma'].iloc[-1]
        long_ma_curr    = df['long_ma'].iloc[-1]
        short_ma_prev   = df['short_ma'].iloc[-2]
        long_ma_prev    = df['long_ma'].iloc[-2]

        # --- Signal ---
        action = "HOLD"
        reason = "No crossover detected"

        if short_ma_prev <= long_ma_prev and short_ma_curr > long_ma_curr:
            action = "BUY"
            reason = f"Golden Cross (SMA{self.short_window} crossed above SMA{self.long_window})"
        elif short_ma_prev >= long_ma_prev and short_ma_curr < long_ma_curr:
            action = "SELL"
            reason = f"Death Cross (SMA{self.short_window} crossed below SMA{self.long_window})"

        # --- Predicted next-candle price via linear regression ---
        predicted_price, predicted_change_pct = self._predict_next(df['close'])

        return {
            "action": action,
            "price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": predicted_change_pct,
            "reason": reason,
            "metadata": {
                f"SMA{self.short_window}": round(short_ma_curr, 2),
                f"SMA{self.long_window}": round(long_ma_curr, 2)
            }
        }

    def _predict_next(self, prices: pd.Series, lookback: int = 20):
        """
        Fit a linear regression over the last `lookback` closing prices
        and extrapolate one step forward.
        """
        try:
            y = prices.dropna().tail(lookback).values
            x = np.arange(len(y))
            # Fit degree-1 polynomial (straight line)
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * len(y) + intercept      # one step beyond last index
            current   = y[-1]
            change_pct = ((predicted - current) / current) * 100
            return round(predicted, 2), round(change_pct, 2)
        except Exception:
            return None, None
