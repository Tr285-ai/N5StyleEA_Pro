from pathlib import Path
import re

p = Path(__file__).resolve().parent.parent / 'trading_strategy.py'
s = p.read_text(encoding='utf-8')
orig = s

# 1) Fix wrong SELL log inside BUY branch: within the BUY block, ensure signal='BUY' and confidence=score
s = re.sub(
    r"(if score >= 0\.6:[\s\S]*?try:\n\s*)log_json\('signal_generated',\s*symbol=symbol,\s*signal='SELL',\s*confidence=1\.0 - score,",
    r"\1log_json('signal_generated', symbol=symbol, signal='BUY', confidence=score,",
    s,
    count=1,
)

# 2) Remove unreachable duplicate BUY logging after an immediate return (defensive cleanup)
s = re.sub(
    r"return out\n\s*try:\n\s*log_json\('signal_generated',\s*symbol=symbol,\s*signal='BUY',[\s\S]*?\)\n\s*except Exception:\n\s*pass\n\s*return out",
    "return out",
    s,
    count=1,
)

# 3) Ensure SELL branch logs SELL with 1.0 - score
s = re.sub(
    r"(if score <= 0\.4:[\s\S]*?try:\n\s*)log_json\('signal_generated',\s*symbol=symbol,\s*signal='BUY',\s*confidence=score,",
    r"\1log_json('signal_generated', symbol=symbol, signal='SELL', confidence=1.0 - score,",
    s,
    count=1,
)

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('trading_strategy.py logging fixed')
else:
    print('No changes to trading_strategy.py')
