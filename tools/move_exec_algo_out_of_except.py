import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'executor.py'
src = p.read_text(encoding='utf-8')

# 1) Remove any in-block assignment of exec_algo
new_src = re.sub(r'^[ \t]*exec_algo\s*=\s*str\(os\.getenv\([^\n]+\)\)\.upper\(\)\s*\n', '', src, flags=re.M)

# 2) Ensure an assignment exists immediately before the if exec_algo in ('TWAP','ICEBERG') line
pattern_if = re.compile(r'^(?P<indent>[ \t]*)if exec_algo in \(\s*[\'"]TWAP[\'"]\s*,\s*[\'"]ICEBERG[\'"]\s*\)\s*:\s*$', re.M)

m = pattern_if.search(new_src)
if m:
    indent = m.group('indent')
    insertion = f"{indent}exec_algo = str(os.getenv('EXEC_ALGO', '')).upper()\n"
    # Insert only if there is not already an exec_algo assignment just above
    start = m.start()
    # Find the start of the previous line
    prev_nl = new_src.rfind('\n', 0, start)
    prev_prev_nl = new_src.rfind('\n', 0, prev_nl)
    prev_line = new_src[prev_prev_nl + 1:prev_nl]
    if 'exec_algo = ' not in prev_line:
        new_src = new_src[:prev_nl+1] + insertion + new_src[prev_nl+1:]

    if new_src != src:
        p.write_text(new_src, encoding='utf-8')
        print('Patched executor.py: moved exec_algo assignment outside except block.')
    else:
        print('No changes applied: executor.py already correct.')
else:
    print('Pattern for exec_algo if-statement not found; no changes.')
