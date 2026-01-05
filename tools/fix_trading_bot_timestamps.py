from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'trading_bot.py'
s = p.read_text(encoding='utf-8')
orig = s

# Ensure timezone is imported
if 'from datetime import datetime, timezone' not in s:
    s = s.replace('from datetime import datetime', 'from datetime import datetime, timezone')

# Replace utcnow occurrences
s = s.replace('datetime.utcnow().isoformat()', 'datetime.now(timezone.utc).isoformat()')
s = s.replace('datetime.utcnow()', 'datetime.now(timezone.utc)')

# Replace _seconds_until_next_candle now assignment
s = s.replace('now = datetime.utcnow()', 'now = datetime.now(timezone.utc)')

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('trading_bot.py timestamps fixed to timezone-aware')
else:
    print('No timestamp changes required in trading_bot.py')
