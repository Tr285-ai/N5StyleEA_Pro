from pathlib import Path
import re

p = Path(r"c:\\N5StyleEA_Pro v15_3\\trading_bot.py")
s = p.read_text(encoding="utf-8")

# If a save is already present near the trade history append, do nothing
if "await self._save_state()" in s:
    # ensure at least one save exists after trade append; otherwise try to insert
    pass

pattern = re.compile(r"^(\s*self\.portfolio\.trade_history\.append\(\{.*\}\))$", re.M)
match = pattern.search(s)
if match:
    line = match.group(1)
    indent = re.match(r"^(\s*)", line).group(1)
    insertion = (
        f"\n{indent}try:\n"
        f"{indent}    await self._save_state()\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
    )
    if insertion.strip() not in s:
        s = s.replace(line, line + insertion, 1)
        p.write_text(s, encoding="utf-8")
        print("Inserted save after trade.")
    else:
        print("Save after trade already present.")
else:
    print("Trade history append line not found; no changes.")
