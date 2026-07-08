"""
Signal Confirmation Filter
---------------------------------
A strategy wrapper that requires N consecutive candles
giving the SAME signal before passing it through.

This eliminates single-candle whipsaws — the #1 cause
of losses in choppy/high-volatility markets.

Usage:
    base = BollingerStrategy()
    confirmed = ConfirmationFilter(base, required_count=2)
    signal = confirmed.analyze(data)
"""

import pandas as pd
from collections import deque
from .base_strategy import BaseStrategy


class ConfirmationFilter(BaseStrategy):
    def __init__(self, wrapped_strategy: BaseStrategy, required_count: int = 2):
        super().__init__(f"{wrapped_strategy.name} [Confirmed×{required_count}]")
        self._inner = wrapped_strategy
        self._required = required_count
        self._history: deque = deque(maxlen=required_count)

    def analyze(self, data: pd.DataFrame) -> dict:
        raw = self._inner.analyze(data)
        action = raw.get("action", "HOLD")

        self._history.append(action)

        # Check if all required consecutive signals match
        if (len(self._history) == self._required and
                len(set(self._history)) == 1 and           # all identical
                list(self._history)[0] in ("BUY", "SELL")):
            # Confirmed — pass through
            confirmed_action = action
        else:
            # Not yet confirmed — downgrade to HOLD
            confirmed_action = "HOLD"

        reason = raw.get("reason", "")
        if confirmed_action == "HOLD" and action in ("BUY", "SELL"):
            reason = f"Waiting for confirmation ({len(self._history)}/{self._required} signals)"

        return {**raw, "action": confirmed_action, "reason": reason}
