import re

with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'r') as f:
    text = f.read()

target = r"                 action = signal\.get\('action', 'HOLD'\)(.*?)# Extract actionable signals from prediction strings"

replacement = r"                 action = signal.get('action', 'HOLD')\n            \n            if isinstance(action, str):\n                 if 'BUY' in action.upper(): action = 'BUY'\n             elif 'SELL' in action.upper(): action = 'SELL'\n\n            # Extract actionable signals from prediction strings"

print("Done")
