import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# ---- Helpers ----
def ensure_import_timezone(text: str) -> str:
    if 'from datetime import datetime, timezone' in text:
        return text
    text = text.replace('from datetime import datetime', 'from datetime import datetime, timezone')
    return text

def replace_utcnow_iso(text: str) -> str:
    text = text.replace('datetime.utcnow().isoformat()', 'datetime.now(timezone.utc).isoformat()')
    return text

def ensure_log_json_import(text: str) -> str:
    if 'from logging_json import log_json' in text:
        return text
    # Add after other imports near top
    lines = text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines[:50]):
        if line.strip().startswith('from tca_utils import') or line.strip().startswith('from risk') or line.strip().startswith('from typing'):
            insert_at = i + 1
    lines.insert(insert_at, 'from logging_json import log_json')
    return '\n'.join(lines)

# ---- Patch executor.py ----
exe_path = root / 'executor.py'
exe_src = exe_path.read_text(encoding='utf-8')
orig_exe = exe_src

# timezone-aware
exe_src = ensure_import_timezone(exe_src)
exe_src = replace_utcnow_iso(exe_src)

# structured logging import
exe_src = ensure_log_json_import(exe_src)

# clean demo TCA: remove child_count erroneous and duplicate session/exchange
exe_src = exe_src.replace("'child_count': len(child_ids),\n", '')
# Remove duplicate session_id/exchange if appears twice in a row
exe_src = exe_src.replace("'session_id': self.session_id,\n                    'exchange': self.exchange_name,\n                    'timestamp': datetime.now(timezone.utc).isoformat(),\n                    'session_id': self.session_id,\n                    'exchange': self.exchange_name,", "'session_id': self.session_id,\n                    'exchange': self.exchange_name,\n                    'timestamp': datetime.now(timezone.utc).isoformat(),")

# Add order_submitted log before placing live order
exe_src = exe_src.replace(
    "performance_monitor.start_timer('order_place')\n        order = await self._exchange_executor.place_market_order(",
    "performance_monitor.start_timer('order_place')\n        try:\n            log_json('order_submitted', mode='LIVE', symbol=str(trade_data.get('symbol')), side=side, amount=float(trade_data.get('quantity') or 0.0))\n        except Exception:\n            pass\n        order = await self._exchange_executor.place_market_order("
)

# Add order_filled log before LIVE order TCA emission
exe_src = exe_src.replace(
    "self._emit_tca_record({\n                    'mode': 'LIVE',",
    "try:\n                log_json('order_filled', mode='LIVE', symbol=str(trade_data.get('symbol')), side=side)\n            except Exception:\n                pass\n                self._emit_tca_record({\n                    'mode': 'LIVE',",
)

# Add order_filled log in algo path before _log_execution
exe_src = exe_src.replace(
    "self._log_execution(out)",
    "try:\n            log_json('order_filled', mode='LIVE', symbol=symbol, side=side, algo=algo, child_count=len(child_ids))\n        except Exception:\n            pass\n        self._log_execution(out)"
)

# Patch Executor.place_market_order retries to log order_retry and order_failed
exe_src = exe_src.replace(
    "performance_monitor.increment_counter('ccxt_order_retries')",
    "performance_monitor.increment_counter('ccxt_order_retries')\n\n                    try:\n                        log_json('order_retry', symbol=symbol, side=side, attempt=attempt+1)\n                    except Exception:\n                        pass"
)
exe_src = exe_src.replace(
    "logger.error(\"Failed to place %s order for %s %s: %s\", side, amount, symbol, str(e))",
    "logger.error(\"Failed to place %s order for %s %s: %s\", side, amount, symbol, str(e))\n\n                    try:\n                        log_json('order_failed', symbol=symbol, side=side, error=str(e))\n                    except Exception:\n                        pass"
)

if exe_src != orig_exe:
    exe_path.write_text(exe_src, encoding='utf-8')
    print('Patched executor.py for timezone and structured logging')
else:
    print('executor.py unchanged')

# ---- Patch trading_bot.py ----
bot_path = root / 'trading_bot.py'
bot_src = bot_path.read_text(encoding='utf-8')
orig_bot = bot_src

# imports
if 'from datetime import datetime, timezone' not in bot_src:
    bot_src = bot_src.replace('from datetime import datetime', 'from datetime import datetime, timezone')
if 'from logging_json import log_json' not in bot_src:
    # place after existing imports
    bot_src = bot_src.replace('from trading_strategy import TradingStrategy', 'from trading_strategy import TradingStrategy\nfrom logging_json import log_json')

# replace utcnow occurrences
bot_src = bot_src.replace('datetime.utcnow()', 'datetime.now(timezone.utc)')
bot_src = bot_src.replace("'timestamp': datetime.utcnow().isoformat()", "'timestamp': datetime.now(timezone.utc).isoformat()")

# log signal in _symbol_loop after generate_signal
bot_src = re.sub(
    r"(signal = await self\.strategy\.generate_signal\(market_data, symbol\)\n\s+)(if signal\.get\('signal'\))",
    r"\1try:\n                log_json('signal_generated', symbol=symbol, signal=lambda s: s.get('signal'), confidence=lambda s: s.get('confidence'), price=lambda s: s.get('price'))\n            except Exception:\n                pass\n            \2",
    bot_src,
    count=1,
)

if bot_src != orig_bot:
    bot_path.write_text(bot_src, encoding='utf-8')
    print('Patched trading_bot.py for timezone and structured logging')
else:
    print('trading_bot.py unchanged')

# ---- Patch trading_strategy.py ----
ts_path = root / 'trading_strategy.py'
ts_src = ts_path.read_text(encoding='utf-8')
orig_ts = ts_src

# imports
ts_src = ts_src.replace('from datetime import datetime', 'from datetime import datetime, timezone')
if 'from logging_json import log_json' not in ts_src:
    ts_src = ts_src.replace('from time_filter import TimeFilter, MarketSession', 'from time_filter import TimeFilter, MarketSession\nfrom logging_json import log_json')

# timezone replacements
ts_src = ts_src.replace('datetime.utcnow()', 'datetime.now(timezone.utc)')
ts_src = ts_src.replace("datetime.utcnow().isoformat()", "datetime.now(timezone.utc).isoformat()")

# BUY branch
ts_src = ts_src.replace(
    "if score >= 0.6:\n                return {",
    "if score >= 0.6:\n                out = {",
)
ts_src = ts_src.replace(
    "'session': session.value,\n                }",
    "'session': session.value,\n                }\n                try:\n                    log_json('signal_generated', symbol=symbol, signal='BUY', confidence=score, price=price, session=session.value)\n                except Exception:\n                    pass\n                return out",
)

# SELL branch
ts_src = ts_src.replace(
    "if score <= 0.4:\n                return {",
    "if score <= 0.4:\n                out = {",
)
ts_src = ts_src.replace(
    "'session': session.value,\n                }",
    "'session': session.value,\n                }\n                try:\n                    log_json('signal_generated', symbol=symbol, signal='SELL', confidence=1.0 - score, price=price, session=session.value)\n                except Exception:\n                    pass\n                return out",
    1,
)

# HOLD path
ts_src = ts_src.replace(
    'return self._hold("no_edge", session=session.value, price=price)',
    "out = self._hold(\"no_edge\", session=session.value, price=price)\n            try:\n                log_json('signal_generated', symbol=symbol, signal=out.get('signal','HOLD'), confidence=out.get('confidence',0.5), price=price, session=session.value, reason=out.get('reason'))\n            except Exception:\n                pass\n            return out",
)

# Error path
ts_src = ts_src.replace(
    'logger.error(f"Error generating signal for {symbol}: {e}")\n            return self._hold("signal_error")',
    'logger.error(f"Error generating signal for {symbol}: {e}")\n            try:\n                log_json(\'signal_error\', symbol=symbol, error=str(e))\n            except Exception:\n                pass\n            return self._hold("signal_error")',
)

if ts_src != orig_ts:
    ts_path.write_text(ts_src, encoding='utf-8')
    print('Patched trading_strategy.py for timezone and structured logging')
else:
    print('trading_strategy.py unchanged')
