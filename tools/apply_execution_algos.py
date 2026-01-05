import re
from pathlib import Path

ROOT = Path(r"c:\\N5StyleEA_Pro v15_3")
TARGET = ROOT / "executor.py"

s = TARGET.read_text(encoding="utf-8")
orig = s

# 1) Ensure extra imports
if "import math" not in s:
    s = s.replace("import asyncio\n\n", "import asyncio\n\nimport math\n\n", 1)

# 2) Insert algo branch in _execute_live_trade after arrival_price determination
if "_execute_live_algo(" not in s:
    # Add helper method before _emit_tca_record or before _log_execution
    anchor = re.search(r"\n\s*def _emit_tca_record\(", s)
    if not anchor:
        anchor = re.search(r"\n\s*def _log_execution\(", s)
    insert_at = anchor.start() if anchor else len(s)
    helper = (
        "\n    async def _execute_live_algo(self, trade_data: Dict[str, Any], side: str, arrival_price: float | None) -> Dict:\n"
        "        algo = str(os.getenv('EXEC_ALGO', '')).upper()\n"
        "        total_amt = float(trade_data.get('quantity') or 0.0)\n"
        "        symbol = str(trade_data.get('symbol'))\n"
        "        child_ids = []\n"
        "        fills = []  # list of (amount, price)\n"
        "        if algo == 'TWAP':\n"
        "            slices = max(1, int(os.getenv('EXEC_TWAP_SLICES', '5')))\n"
        "            duration = max(0.0, float(os.getenv('EXEC_TWAP_DURATION_SEC', str(slices*2))))\n"
        "            per_slice = total_amt / float(slices)\n"
        "            gap = duration / slices if slices > 1 else 0.0\n"
        "            for i in range(slices):\n"
        "                try:\n"
        "                    performance_monitor.start_timer('order_place_child')\n"
        "                    res = await self._exchange_executor.place_market_order(symbol=symbol, side=side, amount=float(per_slice))\n"
        "                    performance_monitor.stop_timer('order_place_child')\n"
        "                    child_ids.append(res.get('id') or res.get('order_id'))\n"
        "                    fp = res.get('average') or res.get('price')\n"
        "                    fills.append((float(per_slice), float(fp) if fp is not None else None))\n"
        "                except Exception:\n"
        "                    performance_monitor.record_error()\n"
        "                if gap > 0 and i < slices - 1:\n"
        "                    await asyncio.sleep(gap)\n"
        "        elif algo == 'ICEBERG':\n"
        "            disp_pct = float(os.getenv('EXEC_ICEBERG_DISPLAY_PCT', '0.2'))\n"
        "            disp_qty = max(1e-8, total_amt * max(0.01, min(disp_pct, 1.0)))\n"
        "            chunks = max(1, int(math.ceil(total_amt / disp_qty)))\n"
        "            for i in range(chunks):\n"
        "                this_amt = float(disp_qty if i < chunks - 1 else total_amt - disp_qty * (chunks - 1))\n"
        "                try:\n"
        "                    performance_monitor.start_timer('order_place_child')\n"
        "                    res = await self._exchange_executor.place_market_order(symbol=symbol, side=side, amount=this_amt)\n"
        "                    performance_monitor.stop_timer('order_place_child')\n"
        "                    child_ids.append(res.get('id') or res.get('order_id'))\n"
        "                    fp = res.get('average') or res.get('price')\n"
        "                    fills.append((this_amt, float(fp) if fp is not None else None))\n"
        "                except Exception:\n"
        "                    performance_monitor.record_error()\n"
        "                await asyncio.sleep(0.5)\n"
        "        # Aggregate result\n"
        "        total_filled = sum(a for a, _ in fills) if fills else 0.0\n"
        "        wavg = None\n"
        "        if fills and all(p is not None for _, p in fills):\n"
        "            denom = sum(a for a, _ in fills)\n"
        "            if denom > 0:\n"
        "                wavg = sum(a*p for a, p in fills if p is not None) / denom\n"
        "        status = 'filled' if total_filled >= total_amt * 0.999 else 'partial'\n"
        "        out = dict(trade_data)\n"
        "        out['status'] = status\n"
        "        out['order_id'] = ','.join([str(x) for x in child_ids if x]) if child_ids else None\n"
        "        if getattr(self, 'metrics_enabled', True):\n"
        "            try:\n"
        "                slippage_bps = compute_slippage_bps(arrival_price, wavg, side) if wavg is not None else None\n"
        "                self._emit_tca_record({\n"
        "                    'mode': 'LIVE',\n"
        "                    'event': 'algo_order',\n"
        "                    'algo': algo,\n"
        "                    'symbol': symbol,\n"
        "                    'side': side,\n"
        "                    'amount': total_amt,\n"
        "                    'arrival_price': arrival_price,\n"
        "                    'fill_price': wavg,\n"
        "                    'slippage_bps': slippage_bps,\n"
        "                    'status': status,\n"
        "                    'child_ids': child_ids,\n"
        "                    'timestamp': datetime.utcnow().isoformat(),\n"
        "                })\n"
        "            except Exception:\n"
        "                pass\n"
        "        self._log_execution(out)\n"
        "        return out\n"
    )
    s = s[:insert_at] + helper + s[insert_at:]

# Insert the branch inside _execute_live_trade after arrival_price computation and before placing single order
if "_execute_live_algo(self, trade_data" in s and "EXEC_ALGO" not in s.split("_execute_live_trade")[1]:
    pat = re.compile(r"(arrival_price\s*=\s*None[\s\S]*?arrival_price\s*=\s*None\s*\r?\n)\s*", re.M)
    # Fallback: find the end of arrival_price try-except block, then insert
    m = re.search(r"\n\s*performance_monitor\.start_timer\('order_place'\)\s*\r?\n", s)
    if m:
        insertion = (
            "        exec_algo = str(os.getenv('EXEC_ALGO', '')).upper()\n"
            "        if exec_algo in ('TWAP','ICEBERG'):\n"
            "            return await self._execute_live_algo(trade_data, side, arrival_price)\n"
        )
        s = s[:m.start()] + insertion + s[m.start():]

if s != orig:
    (ROOT / "executor.py.bak.algos").write_text(orig, encoding="utf-8")
    TARGET.write_text(s, encoding="utf-8")
    print("Applied execution algos (TWAP/ICEBERG) to executor.py")
else:
    print("No changes applied; algos already present")
