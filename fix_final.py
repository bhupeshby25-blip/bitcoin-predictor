import re

with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'r') as f:
    text = f.read()

target = r"            signal = self\.strategy\.action_check\(window\) if hasattr\(self\.strategy, 'action_check'\) else self\.strategy\.analyze\(window\)\n            if not isinstance\(signal, dict\) and isinstance\(signal, str\):\n                 action = signal\n            else:\n                 action = signal\.get\('action', 'HOLD'\)"

replacement = """            signal = self.strategy.action_check(window) if hasattr(self.strategy, 'action_check') else self.strategy.analyze(window)
            action = signal if isinstance(signal, str) else signal.get("action", "HOLD")
            
            if isinstance(action, str):
                action = action.upper()
                if "BUY" in action: action = "BUY"
                elif "SELL" in action: action = "SELL"
                else: action = "HOLD" """

new_text = re.sub(target, replacement, text)

with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'w') as f:
    f.write(new_text)

print(new_text.find("action = action.upper()"))
