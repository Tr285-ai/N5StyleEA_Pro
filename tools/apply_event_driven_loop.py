import re
from pathlib import Path

ROOT = Path(r"c:\\N5StyleEA_Pro v15_3")
TARGET = ROOT / "trading_bot.py"

s = TARGET.read_text(encoding="utf-8")
orig = s

# 1) Insert event-driven branch after state load in run()
if "_run_event_driven(" not in s:
    pat = re.compile(r"(\n\s*await self\._load_state\(\)\s*\r?\n)")
    m = pat.search(s)
    if m:
        insertion = (
            "        if str(os.getenv('USE_EVENT_DRIVEN', '0')).lower() not in ('0', 'false', 'no'):\n"
            "            await self._run_event_driven()\n"
            "            return\n"
        )
        s = s[:m.end()] + insertion + s[m.end():]

# 2) Add helper methods if missing
if "def _run_event_driven(" not in s:
    # Insert before _update_rl_agent definition to ensure class scope
    anchor = re.search(r"\n\s*async def _update_rl_agent\(", s)
    insert_at = anchor.start() if anchor else len(s)
    helpers = (
        "\n    async def _run_event_driven(self) -> None:\n"
        "        self.running = True\n"
        "        try:\n"
        "            tasks = [asyncio.create_task(self._symbol_loop(sym)) for sym in self.symbols]\n"
        "            await asyncio.gather(*tasks)\n"
        "        except asyncio.CancelledError:\n"
        "            pass\n"
        "        finally:\n"
        "            self.running = False\n"
        "\n    async def _symbol_loop(self, symbol: str) -> None:\n"
        "        while self.running:\n"
        "            try:\n"
        "                market_data = await self._get_market_data(symbol)\n"
        "                signal = await self.strategy.generate_signal(market_data, symbol)\n"
        "                if signal.get('signal') and signal['signal'] != 'HOLD':\n"
        "                    await self.execute_trade(symbol, signal)\n"
        "                await self._update_rl_agent(market_data)\n"
        "            except Exception as e:\n"
        "                logger.error(f'Error in symbol loop {symbol}: {e}')\n"
        "            # Sleep until the next candle boundary\n"
        "            try:\n"
        "                delay = self._seconds_until_next_candle(self.timeframe)\n"
        "            except Exception:\n"
        "                delay = 60.0\n"
        "            await asyncio.sleep(max(0.5, float(delay)))\n"
        "\n    def _seconds_until_next_candle(self, timeframe: str) -> float:\n"
        "        now = datetime.utcnow()\n"
        "        tf = (timeframe or '1m').strip().lower()\n"
        "        # Parse timeframe\n"
        "        num = 1\n"
        "        unit = 'm'\n"
        "        try:\n"
        "            if tf[-1] in ('s','m','h','d'):\n"
        "                unit = tf[-1]\n"
        "                num = int(tf[:-1]) if tf[:-1].isdigit() else 1\n"
        "            else:\n"
        "                num = int(tf)\n"
        "                unit = 'm'\n"
        "        except Exception:\n"
        "            num, unit = 1, 'm'\n"
        "        seconds = num * (1 if unit=='s' else 60 if unit=='m' else 3600 if unit=='h' else 86400)\n"
        "        # Compute next boundary\n"
        "        ts = int(now.timestamp())\n"
        "        next_boundary = ((ts // seconds) + 1) * seconds\n"
        "        return float(next_boundary - ts)\n"
    )
    s = s[:insert_at] + helpers + s[insert_at:]

if s != orig:
    (ROOT / "trading_bot.py.bak").write_text(orig, encoding="utf-8")
    TARGET.write_text(s, encoding="utf-8")
    print("Applied event-driven loop to trading_bot.py")
else:
    print("No changes applied; event-driven already present")
