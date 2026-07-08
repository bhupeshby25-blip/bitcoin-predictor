import re

text = \"\"\"            signal = self.strategy.analyze(window)
            if not isinstance(signal, dict) and isinstance(signal, str):
                 action = signal
            else:
                 action = signal.get('action', 'HOLD')\"\"\"

print("STRONGB" in text) 
