import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, account_balance=10000.0, risk_per_trade=0.02):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade

    # ------------------------------------------------------------------ #
    #  Risk Classification
    # ------------------------------------------------------------------ #
    def classify_risk(self, data: pd.DataFrame) -> dict:
        """
        Classify market risk as Low, Medium, or High using ATR/Price ratio.

        ATR/Price < 1.5%  → 🟢 Low
        ATR/Price 1.5–3%  → 🟡 Medium
        ATR/Price > 3%    → 🔴 High

        Returns:
            dict with keys: 'label', 'emoji', 'atr', 'atr_pct', 'description'
        """
        atr = self._calculate_atr(data)
        current_price = data['close'].iloc[-1]
        atr_pct = (atr / current_price) * 100

        if atr_pct < 1.5:
            label, emoji = "Low", "🟢"
            description = "Calm market, tight price swings"
        elif atr_pct < 3.0:
            label, emoji = "Medium", "🟡"
            description = "Moderate volatility, normal conditions"
        else:
            label, emoji = "High", "🔴"
            description = "High volatility, elevated caution needed"

        return {
            "label": label,
            "emoji": emoji,
            "atr": round(atr, 2),
            "atr_pct": round(atr_pct, 2),
            "description": description
        }

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Average True Range over the last `period` candles."""
        df = data.copy().tail(period + 1)
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
        return df['tr'].tail(period).mean()

    # ------------------------------------------------------------------ #
    #  Position Sizing
    # ------------------------------------------------------------------ #
    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> dict:
        """
        Calculate recommended position size based on account risk settings.

        Returns:
            dict with entry, stop loss, risk amount, position size details.
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return {"error": "Prices must be positive"}

        risk_amount = self.account_balance * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            return {"error": "Stop loss cannot equal entry price"}

        position_size = risk_amount / risk_per_unit

        return {
            "entry_price": entry_price,
            "stop_loss": round(stop_loss_price, 2),
            "risk_amount_usd": round(risk_amount, 2),
            "risk_percentage": round(self.risk_per_trade * 100, 1),
            "position_size_units": round(position_size, 6),
            "total_position_value": round(position_size * entry_price, 2)
        }

    def suggest_stop_loss(self, current_price: float, volatility_atr: float, multiplier: float = 2.0):
        """
        Suggest stop-loss levels using ATR-based distance.
        Returns (long_stop, short_stop) tuple.
        """
        long_stop = current_price - (volatility_atr * multiplier)
        short_stop = current_price + (volatility_atr * multiplier)
        return round(long_stop, 2), round(short_stop, 2)
