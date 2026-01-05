import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'executor.py'
src = p.read_text(encoding='utf-8')

pattern = re.compile(r"^(?P<indent>\s*)arrival_price\s*=\s*None\s+exec_algo\s*=\s*(?P<rhs>.+)$", re.M)

if pattern.search(src):
    def _repl(m):
        indent = m.group('indent')
        rhs = m.group('rhs').rstrip()
        return f"{indent}arrival_price = None\n{indent}exec_algo = {rhs}"
    new_src = pattern.sub(_repl, src)
    if new_src != src:
        p.write_text(new_src, encoding='utf-8')
        print('Patched executor.py: split concatenated arrival_price/exec_algo line.')
    else:
        print('No changes applied: substitution produced identical text.')
else:
    print('Pattern not found; no changes made.')
