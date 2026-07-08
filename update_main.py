with open('/Users/bhupesh/Documents/bitcoin-predictor/main.py', 'r') as b: t = b.read().split('\n')
for j in range(len(t)):
     if 'action_emoji = "🟢"' in t[j]:
          t.pop(j)
          t.pop(j)
          t.pop(j)
          
          t.insert(j, '        action_emoji = "🟢" if "BUY" in action else "🔴" if "SELL" in action else "⚫"')
          break

for j in range(len(t)):
    if 'f"🎯 *Target Price*: ${pred_price:,.2f}"' in t[j]:
        t[j] = '            f"🎯 *Analysis Score*: {ml_pred.get(\'conviction_score\', \'—\')} / 100\\n"'
        t.insert(j+1, '            f"📊 *Volatility Base*: {ml_pred.get(\'volatility_context\', \'—\')}%\\n"')
        break
        
with open('/Users/bhupesh/Documents/bitcoin-predictor/main.py', 'w') as b: b.write('\n'.join(t))
