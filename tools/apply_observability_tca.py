import re
import os
from pathlib import Path

ROOT = Path(r"c:\\N5StyleEA_Pro v15_3")
TARGET = ROOT / "executor.py"

s = TARGET.read_text(encoding="utf-8")
orig = s

# 1) Ensure imports
if "from tca_utils import compute_slippage_bps" not in s or "from performance_monitor import performance_monitor" not in s or "import json" not in s:
    pat = re.compile(r"(^\s*import asyncio\s*\r?\n)", re.M)
    m = pat.search(s)
    if m:
        insertion = (
            m.group(1)
            + "import json\nimport random\n\nfrom performance_monitor import performance_monitor\nfrom tca_utils import compute_slippage_bps\n\n"
        )
        s = s[:m.start()] + insertion + s[m.end():]

# 2) TradeExecutor.__init__: add tca fields after self.log_file
pat = re.compile(r"^(\s*)self\.log_file = self\._init_logging\(\)\s*\r?\n", re.M)
mc = pat.search(s)
if mc and "self.tca_file" not in s:
    indent = mc.group(1)
    insertion = (
        f"{indent}self.tca_file = os.path.join(\"logs\", \"tca.jsonl\")\n"
        f"{indent}self.metrics_enabled = str(os.getenv(\"ENABLE_METRICS\", \"1\")).lower() not in {{\"0\", \"false\", \"no\"}}\n"
    )
    s = s[:mc.end()] + insertion + s[mc.end():]

# 3) Replace execute_order body to add metrics
if "performance_monitor.start_timer('execute_order')" not in s:
    pat = re.compile(
        r"(\n\s*async def execute_order\([\s\S]*?\)\s*-> Dict:\s*\r?\n)(\s*\"\"\"Execute a trade order\"\"\"[\s\S]*?)(\n\s*async def _execute_demo_trade\()",
        re.M,
    )
    m = pat.search(s)
    if m:
        new_body = (
            "        \"\"\"Execute a trade order\"\"\"\n"
            "        if not self.connected:\n"
            "            raise RuntimeError(\"Trade executor not connected\")\n\n"
            "        performance_monitor.increment_counter('orders_received')\n"
            "        performance_monitor.start_timer('execute_order')\n"
            "        try:\n"
            "            if not self.risk_manager.validate_order(symbol, side, quantity, price):\n"
            "                raise ValueError(\"Order validation failed\")\n\n"
            "            trade_data = {\n"
            "                'symbol': symbol,\n"
            "                'side': side,\n"
            "                'type': order_type,\n"
            "                'price': price,\n"
            "                'quantity': quantity,\n"
            "                'timestamp': datetime.utcnow().isoformat()\n"
            "            }\n\n"
            "            if self.demo_mode:\n"
            "                result = await self._execute_demo_trade(trade_data)\n"
            "                performance_monitor.increment_counter('orders_demo')\n"
            "            else:\n"
            "                result = await self._execute_live_trade(trade_data)\n"
            "                performance_monitor.increment_counter('orders_live')\n\n"
            "            if isinstance(result, dict) and result.get('status') == 'filled':\n"
            "                performance_monitor.increment_counter('orders_filled')\n"
            "            else:\n"
            "                performance_monitor.increment_counter('orders_submitted')\n"
            "            return result\n"
            "        except Exception:\n"
            "            performance_monitor.record_error()\n"
            "            performance_monitor.increment_counter('orders_failed')\n"
            "            raise\n"
            "        finally:\n"
            "            performance_monitor.stop_timer('execute_order')\n"
        )
        s = s[: m.start(2) ] + new_body + s[m.end(2) :]

# 4) Replace _execute_demo_trade body
if "performance_monitor.start_timer('demo_trade')" not in s:
    pat = re.compile(
        r"(\n\s*async def _execute_demo_trade\([\s\S]*?\)\s*-> Dict:\s*\r?\n)(\s*\"\"\"Execute a demo trade \(simulated\)\"\"\"[\s\S]*?)(\n\s*async def _execute_live_trade\()",
        re.M,
    )
    m = pat.search(s)
    if m:
        new_body = (
            "        \"\"\"Execute a demo trade (simulated)\"\"\"\n"
            "        performance_monitor.start_timer('demo_trade')\n"
            "        logger.info(f\"DEMO TRADE: {trade_data}\")\n"
            "        trade_data['status'] = 'filled'\n"
            "        trade_data['order_id'] = f\"DEMO_{int(time.time())}\"\n"
            "        self._log_execution(trade_data)\n"
            "        placement_ms = performance_monitor.stop_timer('demo_trade')\n"
            "        if getattr(self, 'metrics_enabled', True):\n"
            "            try:\n"
            "                arrival_price = trade_data.get('price')\n"
            "                fill_price = trade_data.get('price')\n"
            "                slippage_bps = compute_slippage_bps(arrival_price, fill_price, str(trade_data.get('side', '')))\n"
            "                self._emit_tca_record({\n"
            "                    'mode': 'DEMO',\n"
            "                    'event': 'order',\n"
            "                    'symbol': trade_data.get('symbol'),\n"
            "                    'side': trade_data.get('side'),\n"
            "                    'amount': trade_data.get('quantity'),\n"
            "                    'arrival_price': arrival_price,\n"
            "                    'fill_price': fill_price,\n"
            "                    'slippage_bps': slippage_bps,\n"
            "                    'status': trade_data.get('status'),\n"
            "                    'order_id': trade_data.get('order_id'),\n"
            "                    'timestamp': datetime.utcnow().isoformat(),\n"
            "                    'latency_ms': {'placement_ms': placement_ms},\n"
            "                })\n"
            "            except Exception:\n"
            "                pass\n"
            "        return trade_data\n"
        )
        s = s[: m.start(2) ] + new_body + s[m.end(2) :]

# 5) Replace _execute_live_trade body
if "performance_monitor.start_timer('order_place')" not in s:
    pat = re.compile(
        r"(\n\s*async def _execute_live_trade\([\s\S]*?\)\s*-> Dict:\s*\r?\n)(\s*\"\"\"Execute a live trade\"\"\"[\s\S]*?)(\n\s*def _log_execution\()",
        re.M,
    )
    m = pat.search(s)
    if m:
        new_body = (
            "        \"\"\"Execute a live trade\"\"\"\n"
            "        if self._exchange_executor is None:\n"
            "            raise RuntimeError(\"Live executor not initialized\")\n\n"
            "        raw_side = str(trade_data.get('side', '')).strip()\n"
            "        upper_side = raw_side.upper()\n"
            "        if upper_side == 'BUY':\n"
            "            side = 'buy'\n"
            "        elif upper_side == 'SELL':\n"
            "            side = 'sell'\n"
            "        else:\n"
            "            side = raw_side.lower()\n\n"
            "        if side not in {'buy', 'sell'}:\n"
            "            raise ValueError(f\"Invalid side for live trade: {raw_side}\")\n\n"
            "        arrival_price = None\n"
            "        try:\n"
            "            performance_monitor.start_timer('ticker_fetch')\n"
            "            tkr = await self._exchange_executor.get_ticker(symbol=str(trade_data.get('symbol')))\n"
            "            performance_monitor.stop_timer('ticker_fetch')\n"
            "            if isinstance(tkr, dict):\n"
            "                ap = tkr.get('last')\n"
            "                arrival_price = float(ap) if ap is not None else None\n"
            "        except Exception:\n"
            "            performance_monitor.record_error()\n"
            "            arrival_price = None\n\n"
            "        performance_monitor.start_timer('order_place')\n"
            "        order = await self._exchange_executor.place_market_order(\n"
            "            symbol=str(trade_data.get('symbol')),\n"
            "            side=side,\n"
            "            amount=float(trade_data.get('quantity') or 0.0),\n"
            "        )\n"
            "        placement_ms = performance_monitor.stop_timer('order_place')\n\n"
            "        out = dict(trade_data)\n"
            "        out['status'] = order.get('status', 'submitted')\n"
            "        out['order_id'] = order.get('id') or order.get('order_id')\n"
            "        self._log_execution(out)\n\n"
            "        if getattr(self, 'metrics_enabled', True):\n"
            "            try:\n"
            "                fill_price = None\n"
            "                cand = order.get('average') or order.get('price')\n"
            "                if cand is not None:\n"
            "                    try:\n"
            "                        fill_price = float(cand)\n"
            "                    except Exception:\n"
            "                        fill_price = None\n"
            "                if fill_price is None:\n"
            "                    try:\n"
            "                        performance_monitor.start_timer('ticker_fetch_post')\n"
            "                        post_tkr = await self._exchange_executor.get_ticker(symbol=str(trade_data.get('symbol')))\n"
            "                        performance_monitor.stop_timer('ticker_fetch_post')\n"
            "                        if isinstance(post_tkr, dict) and post_tkr.get('last') is not None:\n"
            "                            fill_price = float(post_tkr.get('last'))\n"
            "                    except Exception:\n"
            "                        performance_monitor.record_error()\n"
            "                        fill_price = None\n"
            "                slippage_bps = compute_slippage_bps(arrival_price, fill_price, side)\n"
            "                self._emit_tca_record({\n"
            "                    'mode': 'LIVE',\n"
            "                    'event': 'order',\n"
            "                    'symbol': trade_data.get('symbol'),\n"
            "                    'side': side,\n"
            "                    'amount': trade_data.get('quantity'),\n"
            "                    'arrival_price': arrival_price,\n"
            "                    'fill_price': fill_price,\n"
            "                    'slippage_bps': slippage_bps,\n"
            "                    'status': out.get('status'),\n"
            "                    'order_id': out.get('order_id'),\n"
            "                    'timestamp': datetime.utcnow().isoformat(),\n"
            "                    'latency_ms': {'placement_ms': placement_ms},\n"
            "                })\n"
            "            except Exception:\n"
            "                pass\n"
            "        return out\n"
        )
        s = s[: m.start(2) ] + new_body + s[m.end(2) :]

# 6) Add _emit_tca_record if missing
if "def _emit_tca_record(" not in s:
    anchor = re.search(r"\n\s*def _log_execution\([\s\S]*?\n\s*class Executor:\s*", s)
    if anchor:
        insert_at = anchor.start()
        extra = (
            "\n    def _emit_tca_record(self, record: Dict[str, Any]) -> None:\n"
            "        if not getattr(self, 'metrics_enabled', True):\n"
            "            return\n"
            "        try:\n"
            "            os.makedirs(os.path.dirname(self.tca_file), exist_ok=True)\n"
            "            with open(self.tca_file, 'a', encoding='utf-8') as f:\n"
            "                f.write(json.dumps(record, ensure_ascii=False) + '\\n')\n"
            "        except Exception:\n"
            "            pass\n\n"
        )
        s = s[:insert_at] + extra + s[insert_at:]

# 7) Executor.__init__: add ccxt timeout
if "'timeout': int(os.getenv('CCXT_TIMEOUT_MS', '10000'))" not in s:
    pat = re.compile(r"('enableRateLimit': True,\s*\r?\n)(\s+)'options':", re.M)
    s = pat.sub(r"\1\2'timeout': int(os.getenv('CCXT_TIMEOUT_MS', '10000')),\n\2'options':", s, count=1)

# 8) place_market_order retries
if "ccxt_order_retries" not in s:
    pat = re.compile(r"^(\s*)try:\s*\r?\n\s*return await asyncio\.to_thread\([\s\S]*?\)\s*\r?\n\s*except Exception as e:\s*\r?\n\s*logger\.error\([\s\S]*?\)\s*\r?\n\s*raise\s*\r?\n", re.M)
    m = pat.search(s)
    if m:
        indent = m.group(1)
        repl = (
            f"{indent}retries = int(os.getenv('CCXT_MAX_RETRIES', '3'))\n"
            f"{indent}backoff_ms = int(os.getenv('CCXT_RETRY_BACKOFF_MS', '1000'))\n"
            f"{indent}last_err = None\n"
            f"{indent}for attempt in range(max(1, retries)):\n"
            f"{indent}    try:\n"
            f"{indent}        performance_monitor.increment_counter('orders_create_attempts')\n"
            f"{indent}        performance_monitor.start_timer('ccxt_create_order')\n"
            f"{indent}        res = await asyncio.to_thread(\n"
            f"{indent}            self.exchange.create_market_order,\n"
            f"{indent}            symbol,\n"
            f"{indent}            side,\n"
            f"{indent}            amount,\n"
            f"{indent}            params,\n"
            f"{indent}        )\n"
            f"{indent}        performance_monitor.stop_timer('ccxt_create_order')\n"
            f"{indent}        performance_monitor.increment_counter('orders_create_success')\n"
            f"{indent}        return res\n"
            f"{indent}    except Exception as e:\n"
            f"{indent}        last_err = e\n"
            f"{indent}        performance_monitor.record_error()\n"
            f"{indent}        performance_monitor.increment_counter('orders_create_errors')\n"
            f"{indent}        if attempt < retries - 1:\n"
            f"{indent}            performance_monitor.increment_counter('ccxt_order_retries')\n"
            f"{indent}            delay = (backoff_ms * (attempt + 1)) / 1000.0\n"
            f"{indent}            delay += random.uniform(0, 0.25)\n"
            f"{indent}            await asyncio.sleep(delay)\n"
            f"{indent}        else:\n"
            f"{indent}            logger.error(\"Failed to place %s order for %s %s: %s\", side, amount, symbol, str(e))\n"
            f"{indent}            raise last_err\n"
        )
        s = s[:m.start()] + repl + s[m.end():]

# 9) get_ticker timers (skipped here to avoid indentation mismatches). Optional to add later.

if s != orig:
    (ROOT / "executor.py.bak").write_text(orig, encoding="utf-8")
    TARGET.write_text(s, encoding="utf-8")
    print("Applied observability/TCA instrumentation to executor.py")
else:
    print("No changes applied; already instrumented")
