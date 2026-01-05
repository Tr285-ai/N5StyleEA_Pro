import re
from pathlib import Path

ROOT = Path(r"c:\\N5StyleEA_Pro v15_3")
TARGET = ROOT / "trading_bot.py"

s = TARGET.read_text(encoding="utf-8")
orig = s

# 1) Ensure import for MarketDataStreamer
if "from streaming.market_streamer import MarketDataStreamer" not in s:
    # Insert after existing imports block (after trading_strategy import line)
    pat = re.compile(r"^(\s*from trading_strategy import TradingStrategy\s*\r?\n)", re.M)
    m = pat.search(s)
    if m:
        s = s[:m.end()] + "from streaming.market_streamer import MarketDataStreamer\n" + s[m.end():]

# 2) Add self._streamer attribute in __init__ after _exchange init
if "self._streamer = None" not in s:
    pat = re.compile(r"^(\s*)self\._exchange = None\s*\r?\n", re.M)
    m = pat.search(s)
    if m:
        indent = m.group(1)
        s = s[:m.end()] + f"{indent}self._streamer = None\n" + s[m.end():]

# 3) In run(), after _load_state, add optional WS streamer startup BEFORE event-driven toggle
if "USE_WS" not in s or "_run_event_driven" in s:
    pat = re.compile(r"(\n\s*await self\._load_state\(\)\s*\r?\n)")
    m = pat.search(s)
    if m and "_ws_started_flag" not in s:
        insertion = (
            "        # Optional WebSocket streamer (ccxt.pro)\n"
            "        if str(os.getenv('USE_WS', '0')).lower() not in ('0','false','no'):\n"
            "            try:\n"
            "                self._streamer = MarketDataStreamer(self.exchange_name, self.symbols, self.timeframe, self.ohlcv_limit)\n"
            "                await self._streamer.start()\n"
            "                logger.info('WebSocket market streamer started')\n"
            "            except Exception as e:\n"
            "                logger.warning(f'WS streamer unavailable, falling back to REST: {e}')\n"
            "                self._streamer = None\n"
        )
        s = s[:m.end()] + insertion + s[m.end():]

# 4) In run() finally, stop streamer if running
if "await self._save_state()" in s and "_streamer.stop()" not in s:
    pat = re.compile(r"(\n\s*try:\s*\r?\n\s*await self\._save_state\(\)\s*\r?\n\s*except Exception:\s*\r?\n\s*pass\s*\r?\n)")
    m = pat.search(s)
    if m:
        insertion = (
            "            try:\n"
            "                if self._streamer is not None:\n"
            "                    await self._streamer.stop()\n"
            "            except Exception:\n"
            "                pass\n"
        )
        s = s[:m.start()] + insertion + s[m.start():]

# 5) Update _symbol_loop to use streamer when available
if "_symbol_loop" in s and "next_candles(" not in s:
    pat = re.compile(r"(\n\s*async def _symbol_loop\(self, symbol: str\) -> None:\s*\r?\n)([\s\S]*?)(\n\s*def _seconds_until_next_candle)", re.M)
    m = pat.search(s)
    if m:
        body = m.group(2)
        # Replace market_data fetch + sleep logic
        body = re.sub(r"market_data = await self\._get_market_data\(symbol\)",
                      "market_data = await (self._streamer.next_candles(symbol) if self._streamer else self._get_market_data(symbol))",
                      body)
        # Adjust sleep: if streamer is active, do minimal sleep
        body = re.sub(r"await asyncio\.sleep\(max\(0\.5, float\(delay\)\)\)\s*",
                      "            if self._streamer:\n                await asyncio.sleep(0)\n            else:\n                await asyncio.sleep(max(0.5, float(delay)))\n",
                      body)
        s = s[:m.start(2)] + body + s[m.end(2):]

if s != orig:
    (ROOT / "trading_bot.py.bak.ws").write_text(orig, encoding="utf-8")
    TARGET.write_text(s, encoding="utf-8")
    print("Applied WebSocket streaming integration to trading_bot.py")
else:
    print("No changes applied; WS integration already present")
