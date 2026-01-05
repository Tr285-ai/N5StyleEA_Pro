from pathlib import Path
import re

p = Path(__file__).resolve().parent.parent / 'executor.py'
s = p.read_text(encoding='utf-8')
orig = s

# Normalize try/except indentation for LIVE order path
def fix_block(text: str, pattern_src: str, replacement_src: str) -> str:
    pattern = re.compile(pattern_src, re.MULTILINE)
    return pattern.sub(replacement_src, text)

live_pattern = (
    r"(?P<indent>\s*)slippage_bps = compute_slippage_bps\(arrival_price, fill_price, side\)\s*\n"
    r"\s*try:\s*\n\s*log_json\('order_filled',[\s\S]*?\)\s*\n\s*except Exception:\s*\n\s*pass\s*\n\s*self\._emit_tca_record\(\{"
)
live_repl = (
    "\g<indent>slippage_bps = compute_slippage_bps(arrival_price, fill_price, side)\n"
    "\g<indent>try:\n"
    "\g<indent>    log_json('order_filled', mode='LIVE', symbol=str(trade_data.get('symbol')), side=side)\n"
    "\g<indent>except Exception:\n"
    "\g<indent>    pass\n"
    "\g<indent>self._emit_tca_record({"
)

s = fix_block(s, live_pattern, live_repl)

# Normalize try/except indentation for ALGO order path
algo_pattern = (
    r"(?P<indent>\s*)slippage_bps = compute_slippage_bps\(arrival_price, wavg, side\) if wavg is not None else None\s*\n"
    r"\s*try:\s*\n\s*log_json\('order_filled',[\s\S]*?\)\s*\n\s*except Exception:\s*\n\s*pass\s*\n\s*self\._emit_tca_record\(\{"
)
algo_repl = (
    "\g<indent>slippage_bps = compute_slippage_bps(arrival_price, wavg, side) if wavg is not None else None\n"
    "\g<indent>try:\n"
    "\g<indent>    log_json('order_filled', mode='LIVE', symbol=str(trade_data.get('symbol')), side=side)\n"
    "\g<indent>except Exception:\n"
    "\g<indent>    pass\n"
    "\g<indent>self._emit_tca_record({"
)

s = fix_block(s, algo_pattern, algo_repl)

# Ensure all timestamps use timezone-aware now
s = s.replace('datetime.utcnow().isoformat()', 'datetime.now(timezone.utc).isoformat()')

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('executor.py order_filled indentation fixed and timestamps normalized')
else:
    print('No changes to executor.py')
