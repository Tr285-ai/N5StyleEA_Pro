import asyncio
import os
import logging
import random
from typing import Dict, List, Optional

import pandas as pd
from logging_json import log_json
from performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

try:
    import ccxtpro  # type: ignore
except Exception:
    ccxtpro = None  # type: ignore


class MarketDataStreamer:
    def __init__(
        self,
        exchange_name: str,
        symbols: List[str],
        timeframe: str,
        limit: int = 200,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self.exchange_name = exchange_name
        self.symbols = [str(s) for s in symbols]
        self.timeframe = str(timeframe)
        self.limit = int(limit)
        self.api_key = api_key or os.getenv("EXCHANGE_API_KEY")
        self.api_secret = api_secret or os.getenv("EXCHANGE_API_SECRET")

        self.exchange = None
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.buffers: Dict[str, pd.DataFrame] = {}
        self.queues: Dict[str, asyncio.Queue] = {}
        self.reconnect_attempts: int = 0

    async def start(self) -> None:
        if ccxtpro is None:
            raise ImportError("ccxtpro is not installed; cannot use WebSocket streaming")
        ex_class = getattr(ccxtpro, (self.exchange_name or "").lower(), None)
        if ex_class is None:
            raise ValueError(f"Unsupported exchange for streaming: {self.exchange_name}")
        timeout = int(os.getenv("CCXT_TIMEOUT_MS", "10000"))
        self.exchange = ex_class(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "timeout": timeout,
                "options": {"adjustForTimeDifference": True},
            }
        )
        self.running = True
        try:
            log_json('ws_connect', exchange=self.exchange_name, symbols=self.symbols, timeframe=self.timeframe)
        except Exception:
            pass
        try:
            performance_monitor.increment_counter('ws_connects')
        except Exception:
            pass
        for symbol in self.symbols:
            self.buffers[symbol] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])  # noqa
            self.queues[symbol] = asyncio.Queue(maxsize=1)
            t = asyncio.create_task(self._watch_symbol(symbol))
            self.tasks.append(t)

    async def stop(self) -> None:
        self.running = False
        for t in self.tasks:
            try:
                t.cancel()
            except Exception:
                pass
        self.tasks = []
        try:
            if self.exchange is not None:
                await self.exchange.close()
        except Exception:
            pass
        try:
            log_json('ws_shutdown', exchange=self.exchange_name)
        except Exception:
            pass
        try:
            performance_monitor.increment_counter('ws_shutdowns')
        except Exception:
            pass

    async def _watch_symbol(self, symbol: str) -> None:
        assert self.exchange is not None
        while self.running:
            try:
                # Chaos injection for testing resilience
                try:
                    chaos_prob = float(os.getenv('WS_CHAOS_PROB', '0'))
                except Exception:
                    chaos_prob = 0.0
                if chaos_prob > 0 and random.random() < chaos_prob:
                    raise RuntimeError('chaos_injected')
                ohlcv = await self.exchange.watch_ohlcv(symbol, self.timeframe)
                # Expect a list of lists [[ts, o, h, l, c, v], ...]
                if isinstance(ohlcv, list) and ohlcv and isinstance(ohlcv[0], (list, tuple)):
                    rows = ohlcv[-self.limit :]
                    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])  # noqa
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    self.buffers[symbol] = df
                    q = self.queues.get(symbol)
                    if q is not None:
                        if q.full():
                            try:
                                q.get_nowait()
                            except Exception:
                                pass
                        try:
                            q.put_nowait(df.tail(self.limit))
                        except Exception:
                            pass
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"watch_ohlcv error for {symbol}: {e}")
                self.reconnect_attempts += 1
                try:
                    log_json('ws_error', exchange=self.exchange_name, symbol=symbol, error=str(e))
                    log_json('ws_reconnect', exchange=self.exchange_name, symbol=symbol, attempt=self.reconnect_attempts)
                except Exception:
                    pass
                try:
                    performance_monitor.increment_counter('ws_errors')
                    performance_monitor.increment_counter('ws_reconnects')
                except Exception:
                    pass
                # Optional circuit-breaker
                try:
                    max_retries = int(os.getenv('WS_RECONNECT_MAX', '0'))
                except Exception:
                    max_retries = 0
                if max_retries > 0 and self.reconnect_attempts >= max_retries:
                    try:
                        log_json('ws_shutdown', exchange=self.exchange_name, reason='circuit_breaker', symbol=symbol)
                    except Exception:
                        pass
                    try:
                        performance_monitor.increment_counter('ws_shutdowns')
                    except Exception:
                        pass
                    break
                # Backoff with jitter
                try:
                    base_ms = int(os.getenv('WS_RETRY_BACKOFF_MS', os.getenv('CCXT_RETRY_BACKOFF_MS', '1000')))
                except Exception:
                    base_ms = 1000
                try:
                    jitter_ms = int(os.getenv('WS_JITTER_MS', '250'))
                except Exception:
                    jitter_ms = 250
                sleep_s = max(0, base_ms + random.randint(0, max(0, jitter_ms))) / 1000.0
                await asyncio.sleep(sleep_s)

    async def next_candles(self, symbol: str) -> pd.DataFrame:
        q = self.queues.get(symbol)
        if q is None:
            raise RuntimeError(f"No queue for symbol {symbol}")
        df = await q.get()
        return df.tail(self.limit)

    def get_latest(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        df = self.buffers.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])  # noqa
        n = int(limit) if limit is not None else self.limit
        return df.tail(n).copy()
