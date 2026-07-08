"""
Multi-Indicator Confluence Strategy
--------------------------------------
Based on: Combined indicator approach (RSI-2 + EMA-7 + ADX filter)
from research showing BTC outperformance 2012-2025.

This strategy requires MULTIPLE indicators to agree before firing.
Single-indicator strategies are susceptible to false signals;
confluence dramatically improves signal quality.

Conditions for BUY:
  1. RSI(2) < 15         — extreme short-term oversold
  2. Close > EMA(50)     — still in an uptrend
  3. ADX > 20            — sufficient trend present
  4. Close > EMA(7)      — short-term bounce confirmed

Conditions for SELL:
  1. RSI(2) > 85         — extreme short-term overbought
  2. Close < EMA(50)     — uptrend broken
  3. ADX > 20            — sufficient trend present

HOLD otherwise.

Best for: BULL_TREND, BEAR_TREND (replaces standalone MACD/RSI)
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class ConfluenceStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Multi-Indicator Confluence")

    def analyze(self, data: pd.DataFrame) -> dict:
        if len(data) < 55:
            return self._empty(data)

        df = data.copy()
        close = df['close']

        # --- Indicators ---
        rsi2   = self._rsi(close, period=2).iloc[-1]
        ema7   = close.ewm(span=7,  adjust=False).mean().iloc[-1]
        ema50  = close.ewm(span=50, adjust=False).mean().iloc[-1]
        adx    = self._adx(df)
        price  = close.iloc[-1]

        # --- ATR volatility filter ---
        # Only trade when current ATR is within 0.5x to 3x its 20-period average
        # This filters out dead-quiet (no opportunity) and extreme-chaos (random noise)
        atr_now, atr_avg = self._atr_ratio(df)
        atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0
        in_tradeable_vol = 0.5 <= atr_ratio <= 3.0

        action, reason = "HOLD", "No confluence"
        conditions_met = []

        # --- BUY confluence ---
        buy_rsi   = rsi2 < 15
        buy_trend = price > ema50
        buy_adx   = adx > 20
        buy_bounce = price > ema7

        if buy_rsi:
            conditions_met.append("RSI(2)<15 ✅")
        if buy_trend:
            conditions_met.append("Above EMA50 ✅")
        if buy_adx:
            conditions_met.append(f"ADX({adx:.0f})>20 ✅")
        if buy_bounce:
            conditions_met.append("Above EMA7 ✅")

        if buy_rsi and buy_trend and buy_adx and buy_bounce and in_tradeable_vol:
            action = "BUY"
            reason = f"Full confluence BUY: {', '.join(conditions_met)}"

        # --- SELL confluence ---
        sell_rsi   = rsi2 > 85
        sell_trend = price < ema50
        sell_adx   = adx > 20

        if sell_rsi and sell_trend and sell_adx and in_tradeable_vol:
            action = "SELL"
            conditions_met = [
                f"RSI(2)={rsi2:.0f}>85",
                "Below EMA50",
                f"ADX({adx:.0f})>20"
            ]
            reason = f"Full confluence SELL: {', '.join(conditions_met)}"

        if not in_tradeable_vol and action != "HOLD":
            action = "HOLD"
            reason = f"Signal blocked — ATR ratio ({atr_ratio:.2f}) outside tradeable range [0.5, 3.0]"

        predicted_price, predicted_chg = self._predict_next(close)

        return {
            "action":               action,
            "price":                price,
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_chg,
            "reason":               reason,
            "metadata": {
                "RSI(2)":     round(rsi2, 1),
                "EMA7":       round(ema7, 2),
                "EMA50":      round(ema50, 2),
                "ADX":        round(adx, 1),
                "ATR Ratio":  round(atr_ratio, 2)
            }
        }

    def _rsi(self, prices: pd.Series, period: int = 2) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

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

    def _atr_ratio(self, data: pd.DataFrame, period: int = 14, ma_period: int = 20):
        """Returns (current_ATR, average_ATR_over_ma_period)"""
        df = data.copy().tail(period + ma_period + 1)
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['prev_close']).abs(),
            (df['low']  - df['prev_close']).abs(),
        ], axis=1).max(axis=1)
        atr_series = df['tr'].rolling(period).mean()
        atr_now = atr_series.iloc[-1]
        atr_avg = atr_series.tail(ma_period).mean()
        return atr_now, atr_avg

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
