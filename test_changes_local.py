with open('/Users/bhupesh/Documents/bitcoin-predictor/scripts/check_backtest.py','w') as f:
     f.write('''import traceback
try:
    with open('../backtester.py', 'r') as b: t = b.read()
    print("Has MULTI checks?:", "str" in t and "action = signal" in t)
except Exception as e: print(e)''')
