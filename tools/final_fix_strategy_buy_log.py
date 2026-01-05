from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'trading_strategy.py'
s = p.read_text(encoding='utf-8')
orig = s
bad = "log_json('signal_generated', symbol=symbol, signal='SELL', confidence=1.0 - score,"
good = "log_json('signal_generated', symbol=symbol, signal='BUY', confidence=score,"
s = s.replace(bad, good)

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('trading_strategy.py BUY log fixed')
else:
    print('No change needed for trading_strategy.py BUY log')
