from pathlib import Path
import re

p = Path(__file__).resolve().parent.parent / 'main.py'
s = p.read_text(encoding='utf-8')
orig = s

# Ensure Response is imported from fastapi
s = re.sub(r"from fastapi import\s+FastAPI,\s*WebSocket,\s*WebSocketDisconnect\b",
           r"from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response",
           s, count=1)

# Insert /metrics endpoint after root()
pattern = re.compile(r"@app.get\(\"/\"\)\s*\nasync def root\([^)]*\):[\s\S]*?\n\s*return\s+\{[^\n]*\}\s*\n", re.MULTILINE)
match = pattern.search(s)
if match and '/metrics' not in s:
    insert_at = match.end()
    block = (
        "\n\n@app.get(\"/metrics\")\n"
        "async def metrics():\n"
        "    \"\"\"Prometheus metrics endpoint.\"\"\"\n"
        "    try:\n"
        "        from performance_monitor import performance_monitor\n"
        "        body = performance_monitor.render_prometheus()\n"
        "    except Exception:\n"
        "        body = \"\"\n"
        "    return Response(content=body, media_type=\"text/plain; version=0.0.4; charset=utf-8\")\n"
    )
    s = s[:insert_at] + block + s[insert_at:]

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('main.py patched with /metrics endpoint and Response import')
else:
    print('main.py unchanged (metrics endpoint may already exist)')
