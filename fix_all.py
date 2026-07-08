import re

with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'r') as f:
    text = f.read()

target = r"                 action = signal\.get\('action', 'HOLD'\)(.*?)# Extract actionable signals from prediction strings"
replace_head = r"                 action = signal.get('action', 'HOLD')\n            if isinstance(action, str):\n                 if 'BUY' in action.upper():\n                       action = 'BUY'\n                 elif 'SELL' in action.upper():\n                       action = 'SELL'\n                 else:\n                       action = 'HOLD'\n\n            # Extract actionable signals from prediction strings"
text = re.sub(target, replace_head, text, flags=re.DOTALL)

with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'w') as f:
    f.write(text)
print("Replaced.")
