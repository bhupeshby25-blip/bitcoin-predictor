import pandas as pd
import numpy as np
from data_loader import DataLoader
from strategies.simple_ma import SimpleMAStrategy

class Backtester:
    def __init__(self, strategy, initial_capital=10000.0, risk_per_trade=0.02, fee_pct=0.001):
        """
        Args:
            strategy: An instance of a BaseStrategy subclass.
            initial_capital (float): Starting portfolio value in USD.
            risk_per_trade (float): Fraction of capital to risk per trade.
            fee_pct (float): Exchange fee per trade (0.1% = 0.001).
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.fee_pct = fee_pct

    def run(self, data: pd.DataFrame):
        """
        Run the backtest over a historical OHLCV DataFrame.
        Iterates candle by candle, applying the strategy on each step.
        Features:
          - ATR-based entry stop loss
          - ATR trailing stop that rises with price (never falls)
          - Risk-based position sizing (default 2% per trade)
        """
        capital       = self.initial_capital
        position      = 0.0
        entry_price   = 0.0
        trailing_stop = 0.0
        entry_candle  = 0              # candle index when position was opened
        stale_timeout = 10             # exit if not profitable after N candles
        trades        = []
        equity_curve  = []

        min_candles = max(60, getattr(self.strategy, 'long_window', 50))

        for i in range(min_candles, len(data)):
            window = data.iloc[:i+1]
            close  = data['close'].iloc[i]
            atr    = self._atr(window)

            signal = self.strategy.action_check(window) if hasattr(self.strategy, 'action_check') else self.strategy.analyze(window)
            action = signal if isinstance(signal, str) else signal.get("action", "HOLD")

            # Normalize any variant like STRONG_BUY / WEAK_SELL → BUY / SELL / HOLD
            if isinstance(action, str):
                a = action.upper()
                if "BUY" in a:    action = "BUY"
                elif "SELL" in a: action = "SELL"
                else:             action = "HOLD"
            else:
                action = "HOLD"


            # ---- Position management (trailing stop + stale exit) ----
            if position > 0:
                new_trail = close - (2.0 * atr)
                trailing_stop = max(trailing_stop, new_trail)

                # Stale trade exit: if not profitable after N candles, cut it
                candles_held = i - entry_candle
                is_stale = candles_held >= stale_timeout and close <= entry_price

                if close <= trailing_stop or is_stale:
                    exit_type = 'SELL (trail stop)' if close <= trailing_stop else 'SELL (stale exit)'
                    revenue = position * close
                    fee     = revenue * self.fee_pct
                    pnl     = revenue - fee - (entry_price * position)
                    capital += (revenue - fee)
                    trades.append({
                        'type':          exit_type,
                        'date':          data.index[i],
                        'price':         close,
                        'units':         round(position, 6),
                        'pnl':           round(pnl, 2),
                        'pnl_pct':       round((pnl / (entry_price * position)) * 100, 2),
                        'capital_after': round(capital, 2)
                    })
                    position      = 0.0
                    entry_price   = 0.0
                    trailing_stop = 0.0
                    equity_curve.append(capital)
                    continue


            # --- BUY ---
            if action == 'BUY' and position == 0:
                stop_loss     = close - (2.0 * atr)
                stop_loss     = max(stop_loss, close * 0.90)   # floor at -10%
                risk_amount   = capital * self.risk_per_trade
                risk_per_unit = close - stop_loss
                if risk_per_unit <= 0:
                    equity_curve.append(capital)
                    continue
                units = risk_amount / risk_per_unit

                cost = units * close
                fee  = cost * self.fee_pct
                if cost + fee > capital:
                    equity_curve.append(capital)
                    continue

                capital       -= (cost + fee)
                position       = units
                entry_price    = close
                trailing_stop  = stop_loss
                entry_candle   = i

                trades.append({
                    'type':          'BUY',
                    'date':          data.index[i],
                    'price':         close,
                    'units':         round(units, 6),
                    'stop_loss':     round(stop_loss, 2),
                    'capital_after': round(capital, 2)
                })

            # --- Strategy SELL ---
            elif action == 'SELL' and position > 0:
                revenue = position * close
                fee     = revenue * self.fee_pct
                pnl     = revenue - fee - (entry_price * position)
                capital += (revenue - fee)
                trades.append({
                    'type':          'SELL',
                    'date':          data.index[i],
                    'price':         close,
                    'units':         round(position, 6),
                    'pnl':           round(pnl, 2),
                    'pnl_pct':       round((pnl / (entry_price * position)) * 100, 2),
                    'capital_after': round(capital, 2)
                })
                position      = 0.0
                entry_price   = 0.0
                trailing_stop = 0.0

            portfolio_value = capital + (position * close)
            equity_curve.append(portfolio_value)


        # Close any open position at last price
        if position > 0:
            last_price = data['close'].iloc[-1]
            revenue = position * last_price
            fee = revenue * self.fee_pct
            pnl = revenue - fee - (entry_price * position)
            capital += (revenue - fee)
            trades.append({
                'type': 'SELL (end)',
                'date': data.index[-1],
                'price': last_price,
                'units': round(position, 6),
                'pnl': round(pnl, 2),
                'pnl_pct': round((pnl / (entry_price * position)) * 100, 2),
                'capital_after': round(capital, 2)
            })

        return self._compile_results(capital, trades, equity_curve, data)

    def _atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for stop-loss sizing."""
        df = data.copy().tail(period + 1)
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['prev_close']).abs(),
            (df['low']  - df['prev_close']).abs(),
        ], axis=1).max(axis=1)
        return df['tr'].tail(period).mean()

    def _compile_results(self, final_capital, trades, equity_curve, data):
        sell_trades = [t for t in trades if 'pnl' in t]
        wins = [t for t in sell_trades if t['pnl'] > 0]
        losses = [t for t in sell_trades if t['pnl'] <= 0]

        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (len(wins) / len(sell_trades) * 100) if sell_trades else 0

        # Max Drawdown
        equity = pd.Series(equity_curve)
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Buy & Hold comparison
        first_price = data['close'].iloc[50]
        last_price = data['close'].iloc[-1]
        buy_hold_return = ((last_price - first_price) / first_price) * 100

        return {
            'strategy': self.strategy.name,
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_return_pct': round(total_return_pct, 2),
            'buy_hold_return_pct': round(buy_hold_return, 2),
            'total_trades': len(sell_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate_pct': round(win_rate, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'trade_log': trades
        }

    def print_report(self, results):
        print("\n" + "="*50)
        print(f"  📊 BACKTEST RESULTS: {results['strategy']}")
        print("="*50)
        print(f"  Initial Capital :  ${results['initial_capital']:,.2f}")
        print(f"  Final Capital   :  ${results['final_capital']:,.2f}")
        print(f"  Total Return    :  {results['total_return_pct']:+.2f}%")
        print(f"  Buy & Hold      :  {results['buy_hold_return_pct']:+.2f}%")
        print("-"*50)
        print(f"  Total Trades    :  {results['total_trades']}")
        print(f"  Win Rate        :  {results['win_rate_pct']:.1f}%")
        print(f"  Winning Trades  :  {results['winning_trades']}")
        print(f"  Losing Trades   :  {results['losing_trades']}")
        print(f"  Max Drawdown    :  {results['max_drawdown_pct']:.2f}%")
        print("="*50)

        if results['trade_log']:
            print("\n  📋 TRADE LOG:")
            print(f"  {'Type':<12} {'Date':<22} {'Price':>10} {'Units':>10} {'PnL':>10} {'PnL%':>8}")
            print("  " + "-"*76)
            for t in results['trade_log']:
                pnl_str = f"${t.get('pnl', 0):+.2f}" if 'pnl' in t else "---"
                pnl_pct = f"{t.get('pnl_pct', 0):+.2f}%" if 'pnl_pct' in t else "---"
                print(f"  {t['type']:<12} {str(t['date']):<22} ${t['price']:>9,.2f} {t['units']:>10.5f} {pnl_str:>10} {pnl_pct:>8}")
        print()


if __name__ == "__main__":
    print("Fetching historical data (1000 x 1h candles ≈ 41 days)...")
    loader = DataLoader(symbol="BTC/USDT", timeframe="1h")
    data = loader.fetch_ohlcv(limit=1000)

    if data.empty:
        print("❌ No data received. Check your internet connection.")
    else:
        print(f"✅ Got {len(data)} candles from {data.index[0]} to {data.index[-1]}\n")
        strategy = SimpleMAStrategy(short_window=10, long_window=50)
        bt = Backtester(strategy, initial_capital=10000)
        results = bt.run(data)
        bt.print_report(results)
