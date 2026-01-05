import os
import re
from pathlib import Path

ROOT = Path(r"c:\\N5StyleEA_Pro v15_3")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def patch_trading_system():
    p = ROOT / "trading_system.py"
    if not p.exists():
        return "trading_system.py not found"
    s = read_text(p)

    # Wrap fragile imports
    risk_imp = "from risk.advanced_risk import AdvancedRiskManager, RiskMetrics"
    exec_imp = "from execution.order_router import SmartOrderRouter, Order, OrderType"
    mon_imp = "from monitoring.advanced_monitoring import AdvancedMonitor, AlertLevel"

    wrapped_risk = (
        "try:\n"
        "    from risk.advanced_risk import AdvancedRiskManager, RiskMetrics\n"
        "except Exception:\n"
        "    AdvancedRiskManager = None  # type: ignore\n"
        "    RiskMetrics = None  # type: ignore\n"
        "    logging.getLogger(__name__).warning(\"Legacy import failed: risk.advanced_risk; module on deprecation path\")\n"
    )
    wrapped_exec = (
        "try:\n"
        "    from execution.order_router import SmartOrderRouter, Order, OrderType\n"
        "except Exception:\n"
        "    SmartOrderRouter = None  # type: ignore\n"
        "    Order = None  # type: ignore\n"
        "    OrderType = None  # type: ignore\n"
        "    logging.getLogger(__name__).warning(\"Legacy import failed: execution.order_router; module on deprecation path\")\n"
    )
    wrapped_mon = (
        "try:\n"
        "    from monitoring.advanced_monitoring import AdvancedMonitor, AlertLevel\n"
        "except Exception:\n"
        "    AdvancedMonitor = None  # type: ignore\n"
        "    AlertLevel = None  # type: ignore\n"
        "    logging.getLogger(__name__).warning(\"Legacy import failed: monitoring.advanced_monitoring; module on deprecation path\")\n"
    )

    if risk_imp in s:
        s = s.replace(risk_imp, wrapped_risk)
    if exec_imp in s:
        s = s.replace(exec_imp, wrapped_exec)
    if mon_imp in s:
        s = s.replace(mon_imp, wrapped_mon)

    # If wrappers missing (imports not found), insert after 'from enum import Enum'
    if wrapped_risk not in s and "from enum import Enum" in s:
        anchor = "from enum import Enum"
        s = s.replace(anchor, anchor + "\n\n" + wrapped_risk + wrapped_exec + wrapped_mon, 1)

    # Insert legacy adapters before '# Configure logging'
    adapters = (
        "\n\n"
        "def run_legacy_entry(strategy_config: Dict[str, Any]) -> Dict[str, Any]:\n"
        "    logger.warning(\"Legacy entry (trading_system.py) called; routing via TradingBot orchestrator.\")\n"
        "    from legacy_compat import run as legacy_run\n"
        "    return legacy_run(strategy_config)\n\n"
        "async def run_legacy_entry_async(strategy_config: Dict[str, Any]) -> Dict[str, Any]:\n"
        "    logger.warning(\"Legacy entry (trading_system.py, async) called; routing via TradingBot orchestrator.\")\n"
        "    from legacy_compat import run_legacy_strategy as legacy_async\n"
        "    return await legacy_async(strategy_config)\n"
    )
    if "def run_legacy_entry(" not in s and "# Configure logging" in s:
        s = s.replace("\n# Configure logging", adapters + "\n# Configure logging", 1)

    write_text(p, s)
    return "trading_system.py patched"


ess = re.compile(r"^[\t ]*from \.\.(?:.+)$", re.M)


def patch_trading_engine():
    p = ROOT / "trading_engine.py"
    if not p.exists():
        return "trading_engine.py not found"
    s = read_text(p)

    # Replace relative imports block with wrapped imports after 'import logging'
    rel1 = "from ..strategies.expiry_strategies.advanced import AdvancedExpiryStrategy"
    rel2 = "from ..strategies.expiry_strategies.selector import ExpirySelector"
    rel3 = "from .executor import TradeExecutor"

    wrapped = (
        "try:\n"
        "    from ..strategies.expiry_strategies.advanced import AdvancedExpiryStrategy\n"
        "except Exception:\n"
        "    AdvancedExpiryStrategy = None  # type: ignore\n"
        "    logging.getLogger(__name__).warning(\"Legacy import failed: strategies.expiry_strategies.advanced; module on deprecation path\")\n"
        "try:\n"
        "    from ..strategies.expiry_strategies.selector import ExpirySelector\n"
        "except Exception:\n"
        "    ExpirySelector = None  # type: ignore\n"
        "    logging.getLogger(__name__).warning(\"Legacy import failed: strategies.expiry_strategies.selector; module on deprecation path\")\n"
        "try:\n"
        "    from .executor import TradeExecutor\n"
        "except Exception:\n"
        "    try:\n"
        "        from executor import TradeExecutor  # type: ignore\n"
        "    except Exception:\n"
        "        TradeExecutor = None  # type: ignore\n"
        "        logging.getLogger(__name__).warning(\"Legacy import failed: executor; module on deprecation path\")\n"
    )

    if rel1 in s or rel2 in s or rel3 in s:
        s = s.replace(rel1, "").replace(rel2, "").replace(rel3, "")
        s = s.replace("import logging", "import logging\n\n" + wrapped, 1)

    # Add adapters at end
    if "def run_legacy_entry(" not in s:
        s = s.rstrip() + (
            "\n\n\ndef run_legacy_entry(config: Dict[str, Any]) -> Dict[str, Any]:\n"
            "    logger.warning(\"Legacy entry (trading_engine.py) called; routing via TradingBot orchestrator.\")\n"
            "    from legacy_compat import run as legacy_run\n"
            "    return legacy_run(config)\n\n\n"
            "async def run_legacy_entry_async(config: Dict[str, Any]) -> Dict[str, Any]:\n"
            "    logger.warning(\"Legacy entry (trading_engine.py, async) called; routing via TradingBot orchestrator.\")\n"
            "    from legacy_compat import run_legacy_strategy as legacy_async\n"
            "    return await legacy_async(config)\n"
        )

    write_text(p, s)
    return "trading_engine.py patched"


def patch_trading_bot():
    p = ROOT / "trading_bot.py"
    if not p.exists():
        return "trading_bot.py not found"
    s = read_text(p)

    # Ensure imports
    if "import json" not in s:
        s = s.replace("import os\n", "import os\nimport json\n", 1)
    if "import random" not in s:
        s = s.replace("import os\n", "import os\nimport random\n", 1)
        if "import json" not in s:
            s = s.replace("import os\n", "import os\nimport json\nimport random\n", 1)

    # State file init after self._exchange = None
    if "self._state_file" not in s and "self._exchange = None" in s:
        s = s.replace(
            "        self._exchange = None\n",
            "        self._exchange = None\n        self._state_file = os.getenv('BOT_STATE_FILE', 'state/trading_bot_state.json')\n        try:\n            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)\n        except Exception:\n            pass\n",
            1,
        )

    # Load state after executor initialize
    if "await self._load_state()" not in s:
        s = s.replace(
            "            await self.executor.initialize()\n",
            "            await self.executor.initialize()\n            await self._load_state()\n",
            1,
        )

    # Save state in finally
    if "await self._save_state()" not in s:
        s = s.replace(
            "            self.running = False\n",
            "            self.running = False\n            try:\n                await self._save_state()\n            except Exception:\n                pass\n",
            1,
        )

    # Save state after trade history append
    trade_line = "            self.portfolio.trade_history.append({'timestamp': datetime.utcnow(), 'symbol': symbol, 'pnl': 0.0})\n"
    if trade_line in s and "await self._save_state()" not in s:
        s = s.replace(
            trade_line,
            trade_line + "            try:\n                await self._save_state()\n            except Exception:\n                pass\n",
            1,
        )

    # Exchange timeout in ccxt
    s = s.replace(
        "            self._exchange = ex_class({'enableRateLimit': True})\n",
        "            self._exchange = ex_class({'enableRateLimit': True, 'timeout': int(os.getenv('CCXT_TIMEOUT_MS', '10000'))})\n",
        1,
    )

    # Replace fetch_ohlcv block with retry/backoff
    pattern = (
        r"\n\s*ohlcv = await asyncio.to_thread\(\s*self\._exchange\.fetch_ohlcv,\s*symbol,\s*self\.timeframe,\s*None,\s*int\(self\.ohlcv_limit\),\s*\)\s*\n\s*if not ohlcv:\s*\n\s*return None\s*\n"
    )
    if re.search(pattern, s):
        s = re.sub(
            pattern,
            (
                "\n        retries = int(os.getenv('CCXT_MAX_RETRIES', '3'))\n"
                "        backoff_ms = int(os.getenv('CCXT_RETRY_BACKOFF_MS', '1000'))\n"
                "        ohlcv = None\n"
                "        last_err = None\n"
                "        for attempt in range(max(1, retries)):\n"
                "            try:\n"
                "                ohlcv = await asyncio.to_thread(\n"
                "                    self._exchange.fetch_ohlcv,\n"
                "                    symbol,\n"
                "                    self.timeframe,\n"
                "                    None,\n"
                "                    int(self.ohlcv_limit),\n"
                "                )\n"
                "                if ohlcv:\n"
                "                    break\n"
                "            except Exception as e:\n"
                "                last_err = e\n"
                "                delay = (backoff_ms * (attempt + 1)) / 1000.0\n"
                "                delay += random.uniform(0, 0.25)\n"
                "                await asyncio.sleep(delay)\n"
                "        if not ohlcv:\n"
                "            if last_err:\n"
                "                logger.warning(f\"fetch_ohlcv failed after retries for {symbol}: {last_err}\")\n"
                "            return None\n\n"
            ),
            s,
            count=1,
        )

    # Append _load_state/_save_state if missing
    if "def _load_state(" not in s:
        s = s.rstrip() + (
            "\n\n    async def _load_state(self) -> None:\n"
            "        try:\n"
            "            if not self._state_file or not os.path.exists(self._state_file):\n"
            "                return\n"
            "            with open(self._state_file, 'r', encoding='utf-8') as f:\n"
            "                data = json.load(f)\n"
            "            if isinstance(data, dict):\n"
            "                self.balance = float(data.get('balance', self.balance))\n"
            "                positions = data.get('positions') or {}\n"
            "                if isinstance(positions, dict):\n"
            "                    self.positions = {str(k): float(v) for k, v in positions.items()}\n"
            "        except Exception as e:\n"
            "            logger.warning(f\"Failed to load bot state: {e}\")\n"
            "\n    async def _save_state(self) -> None:\n"
            "        try:\n"
            "            state = {\n"
            "                'timestamp': datetime.utcnow().isoformat(),\n"
            "                'balance': float(self.balance),\n"
            "                'positions': {k: float(v) for k, v in self.positions.items()},\n"
            "            }\n"
            "            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)\n"
            "            with open(self._state_file, 'w', encoding='utf-8') as f:\n"
            "                json.dump(state, f)\n"
            "        except Exception as e:\n"
            "            logger.warning(f\"Failed to save bot state: {e}\")\n"
        )

    write_text(p, s)
    return "trading_bot.py patched"


if __name__ == "__main__":
    results = [
        patch_trading_system(),
        patch_trading_engine(),
        patch_trading_bot(),
    ]
    print(" | ".join(results))
