with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'r') as f:
    text = f.read()

target = "            if \"BUY\" in action: action = \"BUY\"\n            elif \"SELL\" in action: action = \"SELL\""
replacement = """            if type(action) == str:
                if 'BUY' in action:
                    action = 'BUY'
                elif 'SELL' in action:
                    action = 'SELL'"""

text = text.replace(target, replacement)

with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'w') as f:
    f.write(text)
