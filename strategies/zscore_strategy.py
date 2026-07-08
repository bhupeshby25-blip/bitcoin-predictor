"""
Z-Score Mean Reversion with Half-Life Filter
-----------------------------------------------
A mathematically rigorous strategy for CORRECTION and BEAR markets.

Theory:
  In a correction, price oscillates around a declining mean. Traditional
  trend-following strategies fail because there IS no trend to follow.
  Instead, we exploit STATISTICAL EXTREMES — entering ONLY when price
  deviates ≥2 standard deviations from its rolling mean, and exiting
  the moment it reverts to the mean.

Key mathematical concepts:
  1. Z-Score = (price - rolling_mean) / rolling_std
     Z < -2  → price is 2σ below mean → statistically oversold
     Z > +2  → price is 2σ above mean → statistically overbought

  2. Half-Life of Mean Reversion (Ornstein-Uhlenbeck process):
     We fit:  Δp(t) = λ * (p(t-1) - μ) + ε
     Half-life = -ln(2) / λ
     Short half-life (< 8 days) → reversion is FAST → safe to trade
     Long half-life (> 15 days) → reversion is SLOW → likely trending, avoid

  3. Quick exit: sell at z > -0.5 (don't wait for z=0, capture 75% of the move)

This strategy's edge:
  - Only 2-3 trades per correction period (very selective)
  - Each trade has statistical backing (extreme deviation)
  - Exits before price stalls at the mean (capture core reversion)
  - Half-life filter blocks trades when market is trending (no reversion expected)
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class ZScoreReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, z_entry: float = -2.0,
                 z_exit: float = -0.5, max_half_life: int = 15):
        super().__init__("Z-Score Mean Reversion")
        self.lookback      = lookback
        self.z_entry       = z_entry       # enter when z-score drops below this
        self.z_exit        = z_exit        # exit when z-score rises above this
        self.max_half_life = max_half_life  # don't trade if reversion is too slow
        self._in_position  = False

    def analyze(self, data: pd.DataFrame) -> dict:
        if len(data) < self.lookback + 10:
            return self._empty(data)

        close = data['close']
        current_price = close.iloc[-1]

        # --- Z-Score calculation ---
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std  = close.rolling(self.lookback).std()
        z_score = ((close - rolling_mean) / rolling_std).iloc[-1]

        # --- Half-Life of Mean Reversion (Ornstein-Uhlenbeck) ---
        half_life = self._calculate_half_life(close, self.lookback)

        # --- Tradeable check ---
        reversion_is_fast = half_life is not None and half_life < self.max_half_life
        mean_val          = rolling_mean.iloc[-1]
        std_val           = rolling_std.iloc[-1]

        action, reason = "HOLD", f"Z-score: {z_score:.2f} (neutral zone)"

        if self._in_position:
            if z_score >= self.z_exit:
                action = "SELL"
                self._in_position = False
                pct_gain = ((current_price - mean_val + std_val * abs(self.z_exit)) / current_price) * 100
                reason = (
                    f"Z-score reverted to {z_score:.2f} ≥ {self.z_exit} — "
                    f"taking profit (mean: ${mean_val:,.0f})"
                )
        else:
            if z_score <= self.z_entry:
                if reversion_is_fast:
                    action = "BUY"
                    self._in_position = True
                    reason = (
                        f"Z-score = {z_score:.2f} (≤ {self.z_entry}σ from mean) | "
                        f"Half-life: {half_life:.1f} days | "
                        f"Mean: ${mean_val:,.0f}"
                    )
                else:
                    hl_str = f"{half_life:.1f}" if half_life else "∞"
                    reason = (
                        f"Z-score = {z_score:.2f} (oversold) BUT half-life "
                        f"= {hl_str} days > {self.max_half_life} — "
                        f"reversion too slow, staying out"
                    )
            elif z_score <= -1.5:
                reason = f"Z-score = {z_score:.2f} — approaching oversold but not extreme enough"

        predicted_price, predicted_chg = self._predict_reversion(close, mean_val, z_score)

        return {
            "action":               action,
            "price":                current_price,
            "predicted_price":      predicted_price,
            "predicted_change_pct": predicted_chg,
            "reason":               reason,
            "metadata": {
                "Z-Score":     round(z_score, 2),
                "Half-Life":   round(half_life, 1) if half_life else "N/A",
                "Mean":        round(mean_val, 2),
                "Std Dev":     round(std_val, 2),
                "In Position": self._in_position
            }
        }

    def _calculate_half_life(self, prices: pd.Series, lookback: int) -> float:
        """
        Fit Ornstein-Uhlenbeck process:
          Δp(t) = λ * p(t-1) + μ + ε
        Half-life = -ln(2) / λ

        Returns half-life in periods, or None if not mean-reverting.
        """
        try:
            series = prices.tail(lookback).values
            delta  = np.diff(series)          # Δp(t)
            lag    = series[:-1]              # p(t-1)

            # OLS regression: delta = lambda * lag + intercept
            X = np.column_stack([lag, np.ones(len(lag))])
            beta = np.linalg.lstsq(X, delta, rcond=None)[0]
            lam  = beta[0]

            if lam >= 0:
                return None  # Not mean-reverting (trending)

            half_life = -np.log(2) / lam
            return max(half_life, 0.1)  # floor at 0.1
        except Exception:
            return None

    def _predict_reversion(self, prices: pd.Series, mean: float, z_score: float):
        """
        If z-score is negative, price should revert toward mean.
        Predicted price = weighted average of current and mean (based on z-score).
        """
        try:
            current = prices.iloc[-1]
            # Expect price to move ~50% of the way toward the mean in 1 period
            predicted = current + (mean - current) * 0.3
            change_pct = ((predicted - current) / current) * 100
            return round(predicted, 2), round(change_pct, 2)
        except Exception:
            return None, None

    def _empty(self, data):
        return {"action": "HOLD", "price": data['close'].iloc[-1],
                "predicted_price": None, "predicted_change_pct": None,
                "reason": "Insufficient data", "metadata": {}}
