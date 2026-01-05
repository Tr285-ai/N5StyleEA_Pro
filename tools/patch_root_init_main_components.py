from pathlib import Path

p = Path(__file__).resolve().parent.parent / '__init__.py'
src = p.read_text(encoding='utf-8')

lines = src.splitlines()
out = []
replaced = False
skip_next = 0
for i, line in enumerate(lines):
    if not replaced and line.strip().startswith('# Import main components'):
        out.append(line)
        out.append('try:')
        out.append('    from .app import app  # type: ignore')
        out.append('    from .core.trading_engine import TradingEngine  # type: ignore')
        out.append('    from .core.api.trade_api import TradeAPI  # type: ignore')
        out.append('except Exception:')
        out.append('    try:')
        out.append('        from app import app  # type: ignore')
        out.append('        from core.trading_engine import TradingEngine  # type: ignore')
        out.append('        from core.api.trade_api import TradeAPI  # type: ignore')
        out.append('    except Exception:')
        out.append('        app = None  # type: ignore')
        out.append('        TradingEngine = None  # type: ignore')
        out.append('        TradeAPI = None  # type: ignore')
        replaced = True
        # Skip original import lines if they immediately follow
        continue
    # Skip any original relative import lines of main components
    if line.strip().startswith('from .app import app') or \
       line.strip().startswith('from .core.trading_engine import TradingEngine') or \
       line.strip().startswith('from .core.api.trade_api import TradeAPI'):
        continue
    out.append(line)

new_src = '\n'.join(out)
if new_src != src:
    p.write_text(new_src, encoding='utf-8')
    print('Patched __init__.py main components import block.')
else:
    print('No changes applied to __init__.py main components block.')
