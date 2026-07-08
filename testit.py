text = """signal = self.strategy.action_check(window)
if r"BUY" in action: action = 'BUY' """
import ast
code = """
import re\n
text = \"\"\"
if 'BUY' in action:\n
\"\"\"\n
import pandas
import numpy\n
print("Executed code block exactly!")
"""
exec(code)
