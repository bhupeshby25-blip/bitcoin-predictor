"""
Volatility Squeeze Breakout Strategy
--------------------------------------
Based on: Bollinger Band squeeze detection (John Carter's TTM Squeeze)

When Bollinger Bands contract inside Keltner Channels, it signals
a period of extreme compression ("squeeze"). When price breaks OUT
of the squeeze, a large move often follows.

Logic:
  1. Detect squeeze: BB upper < KC upper AND BB lower > KC lower
  2. On squeeze release, use momentum (linear regression slope of
     close over 20 bars) to determine direction.
  3. BUY  if squeeze fires + momentum positive + ADX confirming trend
  4. SELL if squeeze fires + momentum negative + ADX confirming trend
  5. HOLD during active squeeze or no squeeze

This strategy dramatically reduces whipsaws because:
 - It only fires at the START of a big move (after compression)
 - False breakouts without squeeze buildup are ignored entirely

Best for: HIGH_VOLATILITY, RANGING (catches the regime TRANSITION)
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class SqueezeStrategy(BaseStrategy):
    def __init__(self, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5, adx_thresh=20):
        super().__init__("Volatility Squeeze Breakout")
        self.bb_period  = bb_period
        self.bb_std     = bb_std
        self.kc_period  = kc_period
        self.kc_mult    = kc_mult
        self.adx_thresh = adx_thresh

    def analyze(self, data: pd.DataFrame) -> dict:
        if len(data) < 60:
            return self._empty(data)

        df = data.copy()

        # --- Bollinger Bands ---
        df['bb_mid']   = df['close'].rolling(self.bb_period).mean()
        df['bb_std']   = df['close'].rolling(self.bb_period).std()
        df['bb_upper'] = df['bb_mid'] + self.bb_std * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - self.bb_std * df['bb_std']

        # --- Keltner Channels (ATR based) ---
        df['kc_mid'] = df['close'].ewm(span=self.kc_period, adjust=False).mean()
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low']  - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = df['tr'].ewm(span=self.kc_period, adjust=False).mean()
        df['kc_upper'] = df['kc_mid'] + self.kc_mult * df['atr']
        df['kc_lower'] = df['kc_mid'] - self.kc_mult * df['atr']

        # --- Squeeze detection ---
        df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

        # Squeeze JUST released = was on, now off
        squeeze_was_on  = df['squeeze_on'].iloc[-2] if len(df) > 1 else False
        squeeze_is_on   = df['squeeze_on'].iloc[-1]
        squeeze_release = squeeze_was_on and not squeeze_is_on

        # --- Momentum (LinReg slope of last 20 closes) ---
        momentum = self._momentum_slope(df['close'])

        # --- ADX filter ---
        adx = self._adx(df)

        current_price = df['close'].iloc[-1]
        action, reason = "HOLD", "No squeeze signal"

        if squeeze_is_on:
            reason = f"🔸 Squeeze active — compression building (BB inside KC)"

        elif squeeze_release:
            if adx > self.adx_thresh:
                if momentum > 0:
                    action = "BUY"
                    reason = f"Squeeze released ↑ | Momentum: +{momentum:.2f} | ADX: {adx:.1f}"
                else:
                    action = "SELL"
                    reason = f"Squeeze released ↓ | Momentum: {momentum:.2f} | ADX: {adx:.1f}"
            else:
                reason = f"Squeeze released but ADX too low ({adx:.1f} < {self.adx_thresh}) — no trade"

        predicted_price, predicted_chg = self._predict_next(df['close'])

        return {
            "action":               action,
            "price":                current_price,
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_chg,
            "reason":               reason,
            "metadata": {
                "Squeeze":     "ON" if squeeze_is_on else ("RELEASED" if squeeze_release else "OFF"),
                "Momentum":    round(momentum, 4),
                "ADX":         round(adx, 1),
            }
        }

    def _momentum_slope(self, prices: pd.Series, lookback: int = 20) -> float:
        """Linear regression slope over last `lookback` closes."""
        y = prices.dropna().tail(lookback).values
        if len(y) < lookback:
            return 0.0
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def _adx(self, data: pd.DataFrame, period: int = 14) -> float:
        df = data.copy().tail(period * 3)
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['prev_close']).abs(),
            (df['low']  - df['prev_close']).abs(),
        ], axis=1).max(axis=1)
        df['+dm'] = df['high'].diff().clip(lower=0)
        df['-dm'] = (-df['low'].diff()).clip(lower=0)
        df.loc[df['+dm'] < df['-dm'], '+dm'] = 0
        df.loc[df['-dm'] < df['+dm'], '-dm'] = 0
        atr   = df['tr'].ewm(alpha=1/period, adjust=False).mean()
        plus  = df['+dm'].ewm(alpha=1/period, adjust=False).mean() / atr * 100
        minus = df['-dm'].ewm(alpha=1/period, adjust=False).mean() / atr * 100
        dx    = ((plus - minus).abs() / (plus + minus) * 100).replace([np.inf, -np.inf], 0)
        return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]

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
