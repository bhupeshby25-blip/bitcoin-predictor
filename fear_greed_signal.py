"""
FearGreedSignal — Empirical Conditional Probability Model
===========================================================
Uses 2019-2026 historical statistics of BTC returns at each
Fear & Greed level to make calibrated directional predictions.

Key empirical findings (from 2,619 daily candles):
  F&G 80-100 (Extreme Greed) → 59% win rate, +2.54% avg 3d return
  F&G 60-70  (Greed)         → 57% win rate, +0.71% avg 3d return
  F&G 30-40  (Fear)          → 45% win rate, -0.37% avg 3d return  ← bearish!
  F&G 10-20  (Extreme Fear)  → 54% day-1 win, but -0.22% 3d avg

COUNTER-INTUITIVE FINDING:
  Extreme Greed = BUY signal (momentum beats contrarian)
  Mid-Fear zone (30-40) = SELL signal (real capitulation not done yet)

The model uses THREE signals:
  1. Bin lookup: historical win rate and avg return for current F&G bin
  2. Momentum: rate of change in F&G over 7 days (improving sentiment = bullish)
  3. Threshold crossing: entering Greed from Fear = strong bullish signal
                         entering Fear from Neutral = bearish warning

Output: a log-return prediction (same scale as ML ensemble models)
        so it can be directly blended into the ensemble.
"""

import numpy as np
import pandas as pd


# Pre-computed statistics from 2019-2026 BTC/USDT daily data
# Format: fng_bin_midpoint → (prob_up, avg_1d_log_return, avg_3d_log_return, n_samples)
_BIN_STATS = {
    5:  (0.509, -0.00225, +0.00496,  53),
    15: (0.544, +0.00244, -0.00215, 217),
    25: (0.498, -0.00033, +0.00223, 456),
    35: (0.449, -0.00199, -0.00370, 276),  # most bearish bin
    45: (0.501, +0.00086, +0.00124, 395),
    55: (0.488, -0.00088, +0.00225, 326),
    65: (0.570, +0.00479, +0.00707, 307),  # strong bullish
    75: (0.499, +0.00121, +0.00366, 435),
    85: (0.591, +0.00651, +0.02535,  88),  # strongest bullish
    95: (0.590, +0.00850, +0.02558,  61),  # strongest bullish
}


class FearGreedSignal:
    """
    A standalone empirical prediction model based purely on Fear & Greed.
    Returns a log-return prediction compatible with the ML ensemble.
    """

    def predict(self, fng_series: pd.Series) -> dict:
        """
        Args:
          fng_series: pd.Series of daily F&G values, indexed by date,
                      sorted oldest→newest.

        Returns:
          dict with:
            log_ret_pred : float  (predicted log return for next day)
            direction    : 'UP' | 'DOWN'
            prob_up      : float  (historical win rate for this F&G level)
            signal       : str    (human-readable reasoning)
        """
        if fng_series.empty or len(fng_series) < 2:
            return self._neutral()

        current_fng = fng_series.iloc[-1]
        if pd.isna(current_fng):
            return self._neutral()

        # --- 1. Bin lookup prediction ---
        bin_mid = self._get_bin_mid(current_fng)
        stats   = _BIN_STATS.get(bin_mid, (0.5, 0.0, 0.0, 0))
        prob_up, avg_1d, avg_3d, n = stats

        # Blend 1d and 3d prediction (weight toward 1d for accuracy)
        base_pred = 0.6 * avg_1d + 0.4 * avg_3d

        # --- 2. F&G Momentum (7-day change) ---
        momentum_pred = 0.0
        if len(fng_series) >= 7:
            fng_7d_change = (fng_series.iloc[-1] - fng_series.iloc[-7]) / 100.0
            # Rising F&G = improving sentiment = slightly bullish
            momentum_pred = fng_7d_change * 0.002

        # --- 3. Threshold crossing signal ---
        crossing_pred = 0.0
        if len(fng_series) >= 3:
            prev  = fng_series.iloc[-2]
            curr  = fng_series.iloc[-1]
            # Crossing INTO greed zone (50→60+): bullish signal
            if prev < 60 and curr >= 60:
                crossing_pred = +0.005
            # Crossing INTO fear zone (50→40-): bearish warning
            elif prev >= 50 and curr < 40:
                crossing_pred = -0.003
            # Exiting extreme fear (15→25): high-probability bounce
            elif prev < 15 and curr >= 15:
                crossing_pred = +0.004

        # --- Combine with sample-size weighting ---
        # Low n_samples → less confident, blend toward 0
        confidence = min(n / 200.0, 1.0)
        total_pred = (base_pred + momentum_pred + crossing_pred) * confidence

        direction = "UP" if total_pred > 0 else "DOWN"

        signal_parts = [f"F&G={current_fng:.0f}"]
        if abs(momentum_pred) > 0.001:
            signal_parts.append(f"momentum={'↑' if momentum_pred > 0 else '↓'}")
        if crossing_pred != 0:
            signal_parts.append(f"crossing={'✅' if crossing_pred > 0 else '⚠️'}")

        return {
            "log_ret_pred": total_pred,
            "direction":    direction,
            "prob_up":      prob_up,
            "signal":       " | ".join(signal_parts),
            "fng_value":    int(current_fng),
            "fng_class":    self._classify(int(current_fng)),
        }

    def _get_bin_mid(self, value: float) -> int:
        """Map F&G value to nearest bin midpoint."""
        clipped = max(0.0, min(99.9, value))
        bin_start = int(clipped // 10) * 10
        return bin_start + 5

    def _classify(self, value: int) -> str:
        if value <= 24:   return "Extreme Fear"
        if value <= 44:   return "Fear"
        if value <= 55:   return "Neutral"
        if value <= 74:   return "Greed"
        return "Extreme Greed"

    def _neutral(self) -> dict:
        return {
            "log_ret_pred": 0.0,
            "direction":    "HOLD",
            "prob_up":      0.5,
            "signal":       "F&G unavailable",
            "fng_value":    50,
            "fng_class":    "Neutral",
        }
