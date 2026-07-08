import urllib.request

# The actual file update block to make the change solid.
with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'r') as b:
    original = b.read().split('\n')
for j in range(len(original)):
    if "action = signal.get('action', 'HOLD')" in original[j]:
        # Inject our normalization logic:
        original.insert(j+1, "            ")
        original.insert(j+2, "            if type(action) is str:")
        original.insert(j+3, "                if 'BUY' in action.upper(): action = 'BUY'")
        original.insert(j+4, "                elif 'SELL' in action.upper(): action = 'SELL'")
        break

with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'w') as b:
    b.write('\n'.join(original))
