import re

file_path = '/Users/bhupesh/Documents/bitcoin-predictor/backtester.py'
with open(file_path, 'r') as f:
    text = f.read()

target = r"            signal = self\.strategy\.action_check\(window\) if hasattr\(self\.strategy, 'action_check'\) else self\.strategy\.analyze\(window\)\n            if not isinstance\(signal, dict\) and isinstance\(signal, str\):\n                 action = signal\n            else:\n                 action = signal\.get\('action', 'HOLD'\)"

replacement = """            signal = self.strategy.action_check(window) if hasattr(self.strategy, 'action_check') else self.strategy.analyze(window)
            if not isinstance(signal, dict) and isinstance(signal, str):
                 action = signal
            else:
                 action = signal.get('action', 'HOLD')
                 
            # Action Translation from Human Confidences to Executable Types 
            if isinstance(action, str):
                if 'BUY' in action.upper():
                    action = "BUY"
                elif 'SELL' in action.upper():
                    action = "SELL"
                else:
                    action = "HOLD"
            else:
                action = "HOLD" """

text = re.sub(target, replacement, text)

with open(file_path, 'w') as f:
    f.write(text)
