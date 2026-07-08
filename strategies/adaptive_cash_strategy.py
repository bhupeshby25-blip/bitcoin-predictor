"""
Adaptive Cash Strategy
--------------------------------------
For BEAR and CORRECTION markets where the mathematically optimal
strategy is: DON'T TRADE (cash bias).

This strategy flips the default from "looking for entries" to
"staying in cash unless conditions are extreme."

Mathematics:
  Expected value of a random long trade in a bear market:
    E[trade] = p(win) * avg_win - p(loss) * avg_loss

  If price is declining at rate r per day, any long position
  starts underwater by r. You need the bounce to exceed r just
  to break even. When r > 0.5%/day, even 60% win rates are
  insufficient because the losses are LARGER than the wins.

  This strategy only trades when the expected recovery exceeds the
  ongoing decline rate — measured by RSI(2) at extreme oversold + volume spike.

Conditions for BUY (all must be true):
  1. RSI(2) < 10           — extreme oversold (bottom 1% of readings)
  2. Volume > 2× average   — capitulation volume (panic selling exhaustion)
  3. Price < Lower Keltner  — extended below volatility envelope
  4. Close > Previous Low   — NOT making a new low (first sign of floor)

Exit: Sell after 3 candles OR at +2% profit (whichever comes first)
      This is a SCALP — don't try to ride it.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class AdaptiveCashStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Adaptive Cash (Bear Optimized)")
        self._bars_held = 0
        self._entry_price = 0.0

    def analyze(self, data: pd.DataFrame) -> dict:
        if len(data) < 30:
            return self._empty(data)

        df = data.copy()
        close = df['close']
        current_price = close.iloc[-1]

        # --- Indicators ---
        rsi2 = self._rsi(close, 2).iloc[-1]

        # Volume spike
        vol_avg = df['volume'].rolling(20).mean().iloc[-1]
        vol_now = df['volume'].iloc[-1]
        vol_spike = vol_now > 2.0 * vol_avg

        # Keltner Channel (lower band)
        ema20 = close.ewm(span=20, adjust=False).mean()
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - close.shift(1)).abs(),
            (df['low']  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=20, adjust=False).mean()
        kc_lower = ema20 - 2.0 * atr

        below_kc = current_price < kc_lower.iloc[-1]

        # Not making a new low
        prev_low = df['low'].iloc[-2]
        not_new_low = close.iloc[-1] > prev_low

        # --- Time-based exit tracking ---
        if self._entry_price > 0:
            self._bars_held += 1
            change_since_entry = ((current_price - self._entry_price) / self._entry_price) * 100

            # Exit: 3-bar timeout OR +2% profit
            if self._bars_held >= 3 or change_since_entry >= 2.0:
                self._entry_price = 0.0
                self._bars_held   = 0
                exit_reason = f"+{change_since_entry:.1f}% profit target" if change_since_entry >= 2.0 else "3-bar timeout"
                return {
                    "action":               "SELL",
                    "price":                current_price,
                    "predicted_price":      None,
                    "predicted_change_pct": None,
                    "reason":               f"Scalp exit: {exit_reason}",
                    "metadata":             {"Bars Held": self._bars_held, "PnL%": round(change_since_entry, 2)}
                }

            return {
                "action":               "HOLD",
                "price":                current_price,
                "predicted_price":      None,
                "predicted_change_pct": None,
                "reason":               f"Holding scalp position ({self._bars_held}/3 bars, {change_since_entry:+.1f}%)",
                "metadata":             {"Bars Held": self._bars_held, "PnL%": round(change_since_entry, 2)}
            }

        # --- Entry logic (all 4 conditions) ---
        action, reason = "HOLD", "Cash bias — no extreme oversold conditions"

        conditions = {
            "RSI(2)<10":        rsi2 < 10,
            "Volume spike":     vol_spike,
            "Below KC lower":   below_kc,
            "Not new low":      not_new_low,
        }
        met = [k for k, v in conditions.items() if v]

        if all(conditions.values()):
            action = "BUY"
            self._entry_price = current_price
            self._bars_held   = 0
            reason = f"Capitulation BUY: {', '.join(met)} | RSI(2)={rsi2:.1f}"
        elif len(met) >= 2:
            reason = f"Partially oversold ({len(met)}/4): {', '.join(met)}"

        predicted_price, predicted_chg = self._predict_next(close)

        return {
            "action":               action,
            "price":                current_price,
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_chg,
            "reason":               reason,
            "metadata": {
                "RSI(2)":        round(rsi2, 1),
                "Vol Spike":     "✅" if vol_spike else "❌",
                "Below KC":      "✅" if below_kc else "❌",
                "Not New Low":   "✅" if not_new_low else "❌",
                "Conditions":    f"{len(met)}/4"
            }
        }

    def _rsi(self, prices: pd.Series, period: int = 2) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _predict_next(self, prices: pd.Series, lookback: int = 10):
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
