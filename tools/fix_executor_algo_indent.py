from pathlib import Path
import re

p = Path(__file__).resolve().parent.parent / 'executor.py'
s = p.read_text(encoding='utf-8')
orig = s

pattern = re.compile(
    r"(?P<indent>\s*)slippage_bps = compute_slippage_bps\(arrival_price, wavg, side\) if wavg is not None else None\s*\n"
    r"\s*try:\s*\n\s*(?P<indent2>\s*)log_json\((?P<args>[^)]*)\)\s*\n\s*except Exception:\s*\n\s*pass\s*\n\s*self\._emit_tca_record\(\{",
    re.MULTILINE,
)

s = pattern.sub(
    "\g<indent>slippage_bps = compute_slippage_bps(arrival_price, wavg, side) if wavg is not None else None\n"
    "\g<indent>try:\n"
    "\g<indent>    log_json(\g<args>)\n"
    "\g<indent>except Exception:\n"
    "\g<indent>    pass\n"
    "\g<indent>self._emit_tca_record({",
    s,
    count=1,
)

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('executor.py algo indentation fixed')
else:
    print('executor.py unchanged (algo indent)')
