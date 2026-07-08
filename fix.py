with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'r') as f:
    text = f.read()

target = r"                action = action\n                if 'BUY' in action.upper(): action = 'BUY'\n"
replace_head = r"                if 'BUY' in action.upper(): action = 'BUY'\n"

text = text.replace("action = action.upper()", "")
with open('/Users/bhupesh/Documents/bitcoin-predictor/backtester.py', 'w') as f:
    f.write(text)
