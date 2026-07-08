import urllib.request
def replace_bt_line():
   try:
       with open('backtester.py', 'r') as b: t = b.read().split('\n')
       for j in range(len(t)):
            if "if type(action) is str:" in t[j]:
                t[j] = "            if isinstance(action, str):"
                t[j+1] = "                 if 'BUY' in action.upper(): action = 'BUY'"
                t[j+2] = "                 elif 'SELL' in action.upper(): action = 'SELL'"
                break
       with open('backtester.py', 'w') as b: b.write('\n'.join(t))
       print("Replaced safely")
   except Exception as e: print(e)
replace_bt_line()
