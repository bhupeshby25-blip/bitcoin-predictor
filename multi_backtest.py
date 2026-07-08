"""
Multi-Horizon Backtest Runner
---------------------------------
Backtests the RegimeAwareStrategy across:
  - Multiple date ranges (bull, bear, sideways periods)
  - Multiple candle timeframes (1d, 1h equivalent via daily)

All fetching uses 1d candles to keep it fast (no rate limiting).
"""

import pandas as pd
import numpy as np
from data_loader import DataLoader
from backtester import Backtester
from strategies.regime_aware_strategy import RegimeAwareStrategy

# ------------------------------------------------------------------ #
#  Scenarios — well-known Bitcoin market eras
# ------------------------------------------------------------------ #
SCENARIOS = [
    {
        "label":     "🐂 2023-2024 Bull Run",
        "start":     "2023-10-01",
        "end":       "2024-03-01",
        "character": "Strong uptrend ($25K → $73K)",
    },
    {
        "label":     "🔥 2024 ATH Euphoria",
        "start":     "2024-09-01",
        "end":       "2025-01-01",
        "character": "Parabolic move ($60K → $100K+)",
    },
    {
        "label":     "📉 2024 Mid-Year Correction",
        "start":     "2024-03-01",
        "end":       "2024-09-01",
        "character": "Post-ATH chop and correction",
    },
    {
        "label":     "↔️  2025 Sideways/Bear",
        "start":     "2024-08-01",
        "end":       "2025-02-01",
        "character": "Range-bound / distribution phase",
    },
    {
        "label":     "📅 Last 6 Months (Recent)",
        "start":     "2025-09-01",
        "end":       "2026-03-01",
        "character": "Most recent market conditions",
    },
]

def run_all():
    loader = DataLoader(symbol="BTC/USDT", timeframe="1d")
    results_summary = []

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  {scenario['label']}")
        print(f"  Period : {scenario['start']} → {scenario['end']}")
        print(f"  Market : {scenario['character']}")
        print(f"{'='*60}")

        data = loader.fetch_ohlcv_range(scenario["start"], scenario["end"])

        if data.empty or len(data) < 60:
            print(f"  ⚠️  Not enough data ({len(data)} candles, need 60+). Skipping.")
            results_summary.append({
                "label": scenario["label"],
                "candles": len(data),
                "error": "Insufficient data"
            })
            continue

        print(f"  Got {len(data)} daily candles\n")

        strategy = RegimeAwareStrategy()
        bt = Backtester(strategy, initial_capital=10000, risk_per_trade=0.02)
        results = bt.run(data)
        bt.print_report(results)

        # Regime breakdown — count how many TRADE signals came from each regime
        regime_counts = _count_regime_signals(data, strategy)
        print("  🌍 Regime Distribution (over period):")
        for regime, count in regime_counts.items():
            bar = "█" * count
            print(f"     {regime:<20}: {bar} ({count} candles)")

        results_summary.append({
            "label":           scenario["label"],
            "candles":         len(data),
            "return_pct":      results["total_return_pct"],
            "buy_hold_pct":    results["buy_hold_return_pct"],
            "win_rate":        results["win_rate_pct"],
            "trades":          results["total_trades"],
            "max_drawdown":    results["max_drawdown_pct"],
        })

    # ---- Final Comparison Table ----
    print(f"\n\n{'='*80}")
    print("  📊 MULTI-HORIZON COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Period':<28} {'Return':>8} {'B&H':>8} {'Win%':>7} {'Trades':>7} {'Drawdown':>10}")
    print(f"  {'-'*70}")
    for r in results_summary:
        if "error" in r:
            print(f"  {r['label']:<28} {'—':>8} {'—':>8} {'—':>7} {'—':>7}  {r['error']}")
        else:
            outcome = "✅" if r["return_pct"] > 0 else "❌"
            print(
                f"  {r['label']:<28} "
                f"{outcome}{r['return_pct']:>+6.1f}% "
                f"{r['buy_hold_pct']:>+8.1f}% "
                f"{r['win_rate']:>6.0f}% "
                f"{r['trades']:>7} "
                f"{r['max_drawdown']:>9.1f}%"
            )
    print(f"{'='*80}\n")


def _count_regime_signals(data: pd.DataFrame, strategy: RegimeAwareStrategy) -> dict:
    """Scan all candles and count how often each regime appears."""
    from strategies.regime_detector import RegimeDetector
    detector = RegimeDetector()
    counts = {"BULL_TREND": 0, "BEAR_TREND": 0, "RANGING": 0, "HIGH_VOLATILITY": 0}
    min_candles = 60
    for i in range(min_candles, len(data)):
        window = data.iloc[:i+1]
        regime = detector.detect(window)["regime"]
        counts[regime] = counts.get(regime, 0) + 1
    return counts


if __name__ == "__main__":
    run_all()
