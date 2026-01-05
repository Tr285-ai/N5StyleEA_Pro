from pathlib import Path

ts = Path(__file__).resolve().parent.parent / 'trading_system.py'
src = ts.read_text(encoding='utf-8')

if 'import pandas as pd' in src:
    print('pandas import already present')
else:
    needle = 'from enum import Enum'
    idx = src.find(needle)
    if idx != -1:
        insert_at = idx + len(needle)
        new_src = src[:insert_at] + '\nimport pandas as pd' + src[insert_at:]
        ts.write_text(new_src, encoding='utf-8')
        print('Inserted pandas import after enum import')
    else:
        # Fallback: insert after top imports block (after last import line near top)
        lines = src.splitlines()
        insert_line = 0
        for i, line in enumerate(lines[:50]):
            if line.startswith('from ') or line.startswith('import '):
                insert_line = i + 1
        lines.insert(insert_line, 'import pandas as pd')
        ts.write_text('\n'.join(lines), encoding='utf-8')
        print('Inserted pandas import using fallback placement')
