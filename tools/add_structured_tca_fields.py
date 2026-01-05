import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'executor.py'
src = p.read_text(encoding='utf-8')
orig = src

# 1) Ensure import uuid is present after import random
if 'import uuid' not in src:
    src = src.replace('\nimport random\n', '\nimport random\nimport uuid\n')

# 2) Ensure self.session_id is assigned in TradeExecutor.__init__
if 'self.session_id = str(uuid.uuid4())' not in src:
    src = src.replace(
        'self._exchange_executor: Optional[Executor] = None\n        \n        logger.info(f"Trade Executor initialized (Mode: {'+"'"+'DEMO' +"'"+' if self.demo_mode else '+"'"+'LIVE'+"'"+'})")\n',
        'self._exchange_executor: Optional[Executor] = None\n        self.session_id = str(uuid.uuid4())\n        \n        logger.info(f"Trade Executor initialized (Mode: {'+"'"+'DEMO' +"'"+' if self.demo_mode else '+"'"+'LIVE'+"'"+'})")\n'
    )

# 3) Inject session_id and exchange into LIVE order TCA record
pattern_order = re.compile(r"self\._emit_tca_record\(\{[\s\S]*?'event': 'order',[\s\S]*?'timestamp': datetime\.utcnow\(\)\.isoformat\(\),\n(?P<indent>\s*)'latency_ms': ")
if not re.search(r"'session_id': self\.session_id", src):
    def repl_order(m):
        indent = m.group('indent') or '                    '
        insertion = f"{indent}'session_id': self.session_id,\n{indent}'exchange': self.exchange_name,\n{indent}"
        return m.group(0).replace("\n" + indent + "'latency_ms': ", "\n" + insertion + "'latency_ms': ")
    src = pattern_order.sub(repl_order, src, count=1)

# 4) Inject child_count, session_id, exchange into algo_order TCA
pattern_algo = re.compile(r"self\._emit_tca_record\(\{[\s\S]*?'event': 'algo_order',[\s\S]*?'child_ids': child_ids,\n(?P<indent>\s*)'timestamp': ")
if not re.search(r"'child_count': len\(child_ids\)", src):
    def repl_algo(m):
        indent = m.group('indent') or '                    '
        insertion = (
            f"{indent}'child_count': len(child_ids),\n"
            f"{indent}'session_id': self.session_id,\n"
            f"{indent}'exchange': self.exchange_name,\n"
            f"{indent}"
        )
        return m.group(0).replace("\n" + indent + "'timestamp': ", "\n" + insertion + "'timestamp': ")
    src = pattern_algo.sub(repl_algo, src, count=1)

if src != orig:
    p.write_text(src, encoding='utf-8')
    print('Structured TCA fields added to executor.py')
else:
    print('No changes applied; executor.py already contains structured fields')
