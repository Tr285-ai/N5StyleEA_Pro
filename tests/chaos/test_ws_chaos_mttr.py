import asyncio
import random
import time
import pathlib
import json
import pytest


class PM:
    def __init__(self):
        self.counters = {}
        self.gauges = {}

    def start_timer(self, name: str):
        return None

    def stop_timer(self, name: str):
        return 0.0

    def increment_counter(self, name: str, value: int = 1):
        self.counters[name] = self.counters.get(name, 0) + int(value)

    def set_gauge(self, name: str, value: float):
        self.gauges[name] = float(value)


@pytest.mark.asyncio
async def test_ws_chaos_mttr_under_budget(monkeypatch):
    monkeypatch.setenv("WS_BREAKER_FAILS", "2")
    monkeypatch.setenv("WS_BREAKER_COOLDOWN_SEC", "0.1")
    # Inject non-zero chaos probability
    monkeypatch.setenv("WS_CHAOS_PROB", "0.3")

    import streaming.market_streamer as sm
    import streaming.ws_breaker_patch as patch

    # Performance monitor in both modules
    pm = PM()
    monkeypatch.setattr(patch, "performance_monitor", pm, raising=True)
    monkeypatch.setattr(sm, "performance_monitor", pm, raising=True)

    # Speed up sleeps
    orig_sleep = asyncio.sleep

    async def fast_sleep(t):
        await orig_sleep(0)

    monkeypatch.setattr(patch.asyncio, "sleep", fast_sleep, raising=True)

    # Fake exchange that sometimes fails
    class FlakyEx:
        async def watch_ohlcv(self, *a, **k):
            # 30% chance of failure, else return one bar
            if random.random() < 0.3:
                raise RuntimeError("ws chaos failure")
            # Return a small OHLCV list
            now = int(time.time() * 1000)
            return [[now, 1, 1, 1, 1, 1]]

    streamer = sm.MarketDataStreamer("binance", ["BTC/USDT"], "1m")
    streamer.exchange = FlakyEx()
    streamer.running = True

    # Track open/close cycles using ws_breaker_open gauge changes
    opens = []
    closes = []

    async def observer():
        last = 0.0
        start_open = None
        deadline = time.monotonic() + 0.5  # run for ~0.5s wall time
        while time.monotonic() < deadline:
            g = pm.gauges.get("ws_breaker_open", 0.0)
            if g == 1.0 and last != 1.0:
                start_open = time.monotonic()
            if g == 0.0 and last == 1.0 and start_open is not None:
                closes.append(time.monotonic() - start_open)
                opens.append(start_open)
                start_open = None
            last = g
            await orig_sleep(0)

    task_stream = asyncio.create_task(streamer._watch_symbol("BTC/USDT"))
    task_obs = asyncio.create_task(observer())
    await orig_sleep(0.6)
    task_stream.cancel()
    task_obs.cancel()
    for t in (task_stream, task_obs):
        try:
            await t
        except BaseException:
            pass

    # Compute MTTR (mean time to recovery) across cycles
    if closes:
        mttr = sum(closes) / max(1, len(closes))
        # RTO budget: < 0.25s given cooldown 0.1s and fast scheduling
        assert mttr < 0.25
    # Write artifact for CI visibility
    outdir = pathlib.Path("artifacts/chaos")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "ws_chaos_mttr.json", "w", encoding="utf-8") as f:
        json.dump({"closes": closes}, f, indent=2)
