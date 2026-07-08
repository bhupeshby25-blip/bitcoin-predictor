"""
Regime-Aware Strategy — v5 BEST-OF-ALL HYBRID
-------------------------------------------------
Uses the historically proven BEST strategy for each regime:

  BULL_TREND      → v4: MACD + SMA (fast entry, +15.2% in bull run)
  HIGH_VOLATILITY → v1: SimpleMA Crossover (was +10% ATH, +12% sideways)
  BEAR_TREND      → NEW: ZScoreReversion + AdaptiveCash (math-driven)
  RANGING         → NEW: ZScoreReversion (statistical extremes only)

Evidence base (from 5-period backtest):
  Bull Run    → v4 (+15.2%) > v1 (+11.7%) > v2 (+7.7%)
  ATH Euphoria → v1 (+10.0%) > v4 (-2.4%)
  Correction  → v3 (-1.5%) > v2 (-6.3%) > v1 (-13.1%)
  Sideways    → v1 (+12.3%) > v2 (+4.3%)
  Recent Bear → v3 (-2.1%) > v4 (-2.3%) > v1 (-6.5%)
"""

import pandas as pd
from .base_strategy import BaseStrategy
from .regime_detector import RegimeDetector
from .macd_strategy import MACDStrategy
from .simple_ma import SimpleMAStrategy
from .zscore_strategy import ZScoreReversionStrategy
from .adaptive_cash_strategy import AdaptiveCashStrategy
from .squeeze_strategy import SqueezeStrategy
from .confirmation_filter import ConfirmationFilter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from predictor import MLPredictor


class RegimeAwareStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Regime-Aware Engine v5")
        self.detector  = RegimeDetector()
        self.predictor = MLPredictor()   # ← shared ML predictor

        self._strategy_map = {
            "BULL_TREND":      [MACDStrategy(), SimpleMAStrategy()],
            "HIGH_VOLATILITY": [SimpleMAStrategy(short_window=7, long_window=21)],
            "BEAR_TREND":      [AdaptiveCashStrategy(), ZScoreReversionStrategy()],
            "RANGING":         [ZScoreReversionStrategy(), SqueezeStrategy()],
        }

    def analyze(self, data: pd.DataFrame) -> dict:
        regime_info = self.detector.detect(data)
        regime      = regime_info["regime"]
        strategies  = self._strategy_map.get(regime, [SimpleMAStrategy()])

        signals = [s.analyze(data) for s in strategies]
        actions = [s["action"] for s in signals]

        # --- Regime-specific consensus rules ---
        if regime == "BULL_TREND":
            # Any BUY → take it (fast entry, don't delay)
            if "BUY" in actions:
                consensus  = "BUY"
                confidence = "High" if actions.count("BUY") == len(actions) else "Medium"
            elif all(a == "SELL" for a in actions):
                consensus, confidence = "SELL", "High"
            else:
                consensus, confidence = "HOLD", "Low"

        elif regime == "HIGH_VOLATILITY":
            # Direct pass-through for SMA crossover (v1 behavior)
            consensus  = actions[0]
            confidence = "High (v1 proven)"

        elif regime in ("BEAR_TREND", "RANGING"):
            # Ultra-conservative: require unanimous agreement for BUY
            if all(a == "BUY" for a in actions):
                consensus, confidence = "BUY", "High (all agree)"
            elif all(a == "SELL" for a in actions):
                consensus, confidence = "SELL", "High (all agree)"
            elif "SELL" in actions:
                # In bear: any SELL signal → respect it
                consensus, confidence = "SELL", "Medium (protective)"
            else:
                consensus, confidence = "HOLD", "Cash bias active"
        else:
            consensus, confidence = "HOLD", "Unknown regime"

        primary = signals[0]
        reasons = " | ".join(
            f"{strategies[i].name}: {signals[i]['action']}"
            for i in range(len(signals))
        )

        # --- ML Price Prediction (overrides naive linreg from sub-strategies) ---
        ml = self.predictor.predict(data)

        return {
            "action":               consensus,
            "price":                primary["price"],
            "predicted_price":      ml["predicted_price"],
            "predicted_change_pct": ml["predicted_change_pct"],
            "prediction_direction": ml.get("direction"),
            "prediction_confidence":ml.get("confidence_pct", 0),
            "model_votes":          ml.get("model_votes", {}),
            "reason":               reasons,
            "confidence":           confidence,
            "regime":               regime_info,
            "sub_signals":          [
                {"strategy": strategies[i].name, "action": signals[i]["action"],
                 "metadata": signals[i]["metadata"]}
                for i in range(len(signals))
            ],
            "metadata":             {
                "Regime":       regime,
                "Confidence":   confidence,
                "ML Direction": ml.get("direction", "—"),
                "ML Confidence":f"{ml.get('confidence_pct', 0):.0f}%",
            }
        }

