"""
Regime Detector — Layer 1 of the strategy hierarchy.

Classifies the current market into one of four regimes using:
  - ADX (Average Directional Index)  → trend strength
  - 200-period MA slope              → trend direction
  - ATR relative to price            → volatility level

Regimes:
  BULL_TREND        — Strong upward trend (ADX > 25, price above 200MA)
  BEAR_TREND        — Strong downward trend (ADX > 25, price below 200MA)
  RANGING           — No clear trend (ADX < 20)
  HIGH_VOLATILITY   — Extreme ATR spike regardless of trend
"""

import pandas as pd
import numpy as np


class RegimeDetector:
    HIGH_VOLATILITY_ATR_PCT = 3.5   # ATR/price > 3.5% = high vol
    TREND_ADX_THRESHOLD     = 25    # ADX > 25 = trending
    RANGING_ADX_THRESHOLD   = 20    # ADX < 20 = ranging

    def detect(self, data: pd.DataFrame) -> dict:
        """
        Detect market regime from OHLCV data.

        Returns:
            dict:
                'regime'      : str  — 'BULL_TREND' | 'BEAR_TREND' | 'RANGING' | 'HIGH_VOLATILITY'
                'emoji'       : str
                'adx'         : float
                'ma200_slope' : float
                'atr_pct'     : float
                'description' : str
        """
        if len(data) < 60:
            return self._regime_result("RANGING", data)

        adx      = self._calculate_adx(data)
        ma50     = data['close'].rolling(50).mean()
        slope    = self._ma_slope(ma50)
        atr_pct  = self._atr_pct(data)
        price    = data['close'].iloc[-1]
        above50  = price > ma50.iloc[-1]

        # High volatility overrides everything
        if atr_pct > self.HIGH_VOLATILITY_ATR_PCT:
            regime = "HIGH_VOLATILITY"

        # Strong trend
        elif adx > self.TREND_ADX_THRESHOLD:
            regime = "BULL_TREND" if above50 else "BEAR_TREND"

        # No clear trend
        else:
            regime = "RANGING"

        return self._regime_result(regime, data, adx=adx, slope=slope, atr_pct=atr_pct)

    # ------------------------------------------------------------------ #
    #  Indicator calculations
    # ------------------------------------------------------------------ #
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        df = data.copy().tail(period * 3)
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = (
            pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['prev_close']).abs(),
                (df['low']  - df['prev_close']).abs()
            ], axis=1).max(axis=1)
        )
        df['+dm'] = df['high'].diff().clip(lower=0)
        df['-dm'] = (-df['low'].diff()).clip(lower=0)
        df.loc[df['+dm'] < df['-dm'], '+dm'] = 0
        df.loc[df['-dm'] < df['+dm'], '-dm'] = 0

        atr   = df['tr'].ewm(alpha=1/period, adjust=False).mean()
        plus  = df['+dm'].ewm(alpha=1/period, adjust=False).mean() / atr * 100
        minus = df['-dm'].ewm(alpha=1/period, adjust=False).mean() / atr * 100
        dx    = ((plus - minus).abs() / (plus + minus) * 100).replace([np.inf, -np.inf], 0)
        return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]

    def _ma_slope(self, ma: pd.Series, lookback: int = 5) -> float:
        """Slope of the MA over last N periods as a % change."""
        tail = ma.dropna().tail(lookback)
        if len(tail) < 2:
            return 0.0
        return ((tail.iloc[-1] - tail.iloc[0]) / tail.iloc[0]) * 100

    def _atr_pct(self, data: pd.DataFrame, period: int = 14) -> float:
        df = data.copy().tail(period + 1)
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = (
            pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['prev_close']).abs(),
                (df['low']  - df['prev_close']).abs()
            ], axis=1).max(axis=1)
        )
        atr = df['tr'].tail(period).mean()
        return round((atr / data['close'].iloc[-1]) * 100, 2)

    # ------------------------------------------------------------------ #
    #  Result builder
    # ------------------------------------------------------------------ #
    def _regime_result(self, regime: str, data: pd.DataFrame, **kwargs) -> dict:
        labels = {
            "BULL_TREND":       ("📈 Bull Trend",       "Strong upward momentum — trend-following strategies apply"),
            "BEAR_TREND":       ("📉 Bear Trend",       "Strong downward momentum — defensive stance, look for bounces"),
            "RANGING":          ("↔️  Ranging Market",   "No clear direction — mean-reversion strategies apply"),
            "HIGH_VOLATILITY":  ("⚡ High Volatility",  "Extreme price swings — breakout strategies or reduced sizing"),
        }
        emoji_label, description = labels.get(regime, ("❓ Unknown", ""))
        return {
            "regime":      regime,
            "emoji_label": emoji_label,
            "description": description,
            "adx":         round(kwargs.get("adx", 0.0), 1),
            "ma200_slope": round(kwargs.get("slope", 0.0), 3),
            "atr_pct":     kwargs.get("atr_pct", 0.0),
        }
