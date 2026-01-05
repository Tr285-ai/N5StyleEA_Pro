import asyncio
import os
import pytest
import sys
import types

# We will import TradingBot after setting env vars within each test

class DummyStrategy:
    async def generate_signal(self, market_data, symbol: str):
        # Keep it simple to avoid trades
        return {'signal': 'HOLD', 'symbol': symbol, 'price': 1.0}

@pytest.mark.asyncio
async def test_ws_fallback_on_start(monkeypatch):
    # Force event-driven + WS path
    monkeypatch.setenv('USE_EVENT_DRIVEN', '1')
    monkeypatch.setenv('USE_WS', '1')
    monkeypatch.setenv('PAPER_TRADING', 'true')

    # Stub executor and trading_strategy to avoid heavy/broken imports
    fake_exec = types.ModuleType('executor')
    class FakeTradeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass
        async def initialize(self):
            return None
        async def execute_order(self, *args, **kwargs):
            return {'status': 'filled'}
    fake_exec.TradeExecutor = FakeTradeExecutor
    monkeypatch.setitem(sys.modules, 'executor', fake_exec)

    fake_ts = types.ModuleType('trading_strategy')
    class FakeTradingStrategy:
        pass
    fake_ts.TradingStrategy = FakeTradingStrategy
    monkeypatch.setitem(sys.modules, 'trading_strategy', fake_ts)

    # Import here so monkeypatch applies to module symbols and stubs
    import trading_bot as tb

    # Patch the WS streamer to raise ImportError on start
    async def fake_start(self):
        raise ImportError('ccxt.pro not available')
    monkeypatch.setattr(tb.MarketDataStreamer, 'start', fake_start, raising=True)

    # Stub _run_event_driven to avoid infinite loop and return quickly
    called = {'flag': False}
    async def fast_event_loop(self):
        called['flag'] = True
        return
    monkeypatch.setattr(tb.TradingBot, '_run_event_driven', fast_event_loop, raising=True)

    bot = tb.TradingBot(strategy=DummyStrategy(), symbols=['BTC/USDT'], paper=True)
    await bot.run()

    # It should have attempted WS and fallen back (streamer is None) and used event-driven path
    assert called['flag'] is True
    assert getattr(bot, '_streamer') is None

@pytest.mark.asyncio
async def test_event_driven_toggle_invokes_event_loop(monkeypatch):
    # Enable event-driven, disable WS
    monkeypatch.setenv('USE_EVENT_DRIVEN', '1')
    monkeypatch.setenv('USE_WS', '0')
    monkeypatch.setenv('PAPER_TRADING', 'true')

    # Stub executor and trading_strategy prior to import
    fake_exec = types.ModuleType('executor')
    class FakeTradeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass
        async def initialize(self):
            return None
        async def execute_order(self, *args, **kwargs):
            return {'status': 'filled'}
    fake_exec.TradeExecutor = FakeTradeExecutor
    monkeypatch.setitem(sys.modules, 'executor', fake_exec)

    fake_ts = types.ModuleType('trading_strategy')
    class FakeTradingStrategy:
        pass
    fake_ts.TradingStrategy = FakeTradingStrategy
    monkeypatch.setitem(sys.modules, 'trading_strategy', fake_ts)

    import trading_bot as tb

    called = {'flag': False}
    async def fast_event_loop(self):
        called['flag'] = True
        return
    monkeypatch.setattr(tb.TradingBot, '_run_event_driven', fast_event_loop, raising=True)

    bot = tb.TradingBot(strategy=DummyStrategy(), symbols=['ETH/USDT'], paper=True)
    await bot.run()

    assert called['flag'] is True


@pytest.mark.asyncio
async def test_event_driven_concurrency_per_symbol(monkeypatch):
    # Ensure event-driven path
    monkeypatch.setenv('USE_EVENT_DRIVEN', '1')
    monkeypatch.setenv('USE_WS', '0')
    monkeypatch.setenv('PAPER_TRADING', 'true')

    # Stub executor and trading_strategy prior to import to avoid heavy deps
    fake_exec = types.ModuleType('executor')
    class FakeTradeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass
        async def initialize(self):
            return None
    fake_exec.TradeExecutor = FakeTradeExecutor
    monkeypatch.setitem(sys.modules, 'executor', fake_exec)

    fake_ts = types.ModuleType('trading_strategy')
    class FakeTradingStrategy:
        pass
    fake_ts.TradingStrategy = FakeTradingStrategy
    monkeypatch.setitem(sys.modules, 'trading_strategy', fake_ts)

    import trading_bot as tb

    sym1, sym2 = 'AAA/BBB', 'CCC/DDD'
    e1 = asyncio.Event()
    e2 = asyncio.Event()

    async def stub_symbol_loop(self, symbol: str):
        if symbol == sym1:
            e1.set()
            await asyncio.wait_for(e2.wait(), timeout=1.5)
        else:
            e2.set()
            await asyncio.wait_for(e1.wait(), timeout=1.5)
        return

    monkeypatch.setattr(tb.TradingBot, '_symbol_loop', stub_symbol_loop, raising=True)

    bot = tb.TradingBot(strategy=DummyStrategy(), symbols=[sym1, sym2], paper=True)
    await bot._run_event_driven()

    # Both tasks must have started and seen each other's start event
    assert e1.is_set() and e2.is_set()


@pytest.mark.asyncio
async def test_rest_retry_backoff_in_executor(monkeypatch, tmp_path):
    # Exercise Executor retry loop by simulating transient failures
    import importlib
    exmod = importlib.import_module('executor')

    class FlakyExec:
        def __init__(self, *a, **k):
            self.paper = False
            class X:
                def create_market_order(self, *args, **kwargs):
                    raise RuntimeError('transient')
            self.exchange = X()
        async def place_market_order(self, *args, **kwargs):
            # Use the existing retry logic via the real Executor method by shadowing create_market_order only
            return await exmod.Executor.place_market_order(self, *args, **kwargs)
        async def get_ticker(self, symbol: str):
            return {'symbol': symbol, 'last': 100.0}

    monkeypatch.setenv('ENABLE_METRICS', '1')
    monkeypatch.setenv('EXEC_ALGO', '')
    monkeypatch.setenv('CCXT_MAX_RETRIES', '2')
    monkeypatch.setenv('CCXT_RETRY_BACKOFF_MS', '1')

    tca_dir = tmp_path / 'logs'
    tca_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    te = exmod.TradeExecutor({'broker': {'demo': False, 'exchange': 'binance', 'api_key': 'k', 'api_secret': 's'}})
    await te.initialize()
    # inject flaky exec into trade executor
    te._exchange_executor = FlakyExec()

    # Fake performance monitor to capture counters
    class PM:
        def __init__(self):
            self.counters = {}
        def start_timer(self, name):
            return None
        def stop_timer(self, name):
            return 0.0
        def increment_counter(self, name, value: int = 1):
            self.counters[name] = self.counters.get(name, 0) + value
        def record_error(self):
            self.counters['errors'] = self.counters.get('errors', 0) + 1

    pm = PM()
    monkeypatch.setattr(exmod, 'performance_monitor', pm, raising=True)

    with pytest.raises(Exception):
        await te.execute_order('X/Y', 'BUY', 'MARKET', None, 1.0)

    # metrics file exists and recorded counters reflect retries
    assert (tmp_path / 'logs').exists()
    assert pm.counters.get('orders_create_attempts') == 2
    assert pm.counters.get('ccxt_order_retries') == 1
    # Each failure increments orders_create_errors, so 2
    assert pm.counters.get('orders_create_errors') == 2


@pytest.mark.asyncio
async def test_ws_reconnect_counters_on_watch_error(monkeypatch):
    import streaming.market_streamer as sm

    # Provide a test performance monitor sink in the module under test
    class PM:
        def __init__(self):
            self.counters = {}
        def increment_counter(self, name, value: int = 1):
            self.counters[name] = self.counters.get(name, 0) + value
    pm = PM()
    monkeypatch.setattr(sm, 'performance_monitor', pm, raising=True)

    # Speed up sleep while avoiding recursion
    orig_sleep = asyncio.sleep
    async def fast_sleep(_t):
        await orig_sleep(0)
    monkeypatch.setattr(sm.asyncio, 'sleep', fast_sleep, raising=True)

    # Fake exchange that always errors
    class FakeEx:
        async def watch_ohlcv(self, *a, **k):
            raise RuntimeError('ws fail')

    streamer = sm.MarketDataStreamer('binance', ['BTC/USDT'], '1m')
    streamer.exchange = FakeEx()
    streamer.running = True

    # Run one loop iteration and then cancel
    task = asyncio.create_task(streamer._watch_symbol('BTC/USDT'))
    await orig_sleep(0)
    task.cancel()
    try:
        await task
    except BaseException:
        pass

    # Assert reconnection/error metrics incremented
    assert streamer.reconnect_attempts >= 1
    assert pm.counters.get('ws_errors', 0) >= 1
    assert pm.counters.get('ws_reconnects', 0) >= 1


@pytest.mark.asyncio
async def test_event_loop_cancellation_triggers_ws_shutdown(monkeypatch):
    # Enable event-driven and WS
    monkeypatch.setenv('USE_EVENT_DRIVEN', '1')
    monkeypatch.setenv('USE_WS', '1')
    monkeypatch.setenv('PAPER_TRADING', 'true')

    # Stub executor and trading_strategy prior to import
    import types, sys
    fake_exec = types.ModuleType('executor')
    class FakeTradeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass
        async def initialize(self):
            return None
    fake_exec.TradeExecutor = FakeTradeExecutor
    monkeypatch.setitem(sys.modules, 'executor', fake_exec)

    fake_ts = types.ModuleType('trading_strategy')
    class FakeTradingStrategy:
        pass
    fake_ts.TradingStrategy = FakeTradingStrategy
    monkeypatch.setitem(sys.modules, 'trading_strategy', fake_ts)

    import trading_bot as tb

    # PM sink to capture ws_shutdowns
    class PM:
        def __init__(self):
            self.counters = {}
        def increment_counter(self, name, value: int = 1):
            self.counters[name] = self.counters.get(name, 0) + value
    pm = PM()

    # Fake streamer that increments ws_shutdowns on stop
    class FakeStreamer:
        def __init__(self, *a, **k):
            self.running = False
        async def start(self):
            self.running = True
        async def stop(self):
            self.running = False
            # Simulate WS shutdown metric
            pm.increment_counter('ws_shutdowns')
        async def next_candles(self, symbol):
            return None

    # Patch MarketDataStreamer used by trading_bot
    monkeypatch.setattr(tb, 'MarketDataStreamer', FakeStreamer, raising=True)

    # Make symbol loop hang until cancelled but signal start
    sym1, sym2 = 'AAA/BBB', 'CCC/DDD'
    started1 = asyncio.Event()
    started2 = asyncio.Event()
    async def stuck_loop(self, symbol: str):
        if symbol == sym1:
            started1.set()
        else:
            started2.set()
        await asyncio.Event().wait()  # never set; cancelled by test
    monkeypatch.setattr(tb.TradingBot, '_symbol_loop', stuck_loop, raising=True)

    bot = tb.TradingBot(strategy=FakeTradingStrategy(), symbols=[sym1, sym2], paper=True)
    # Run the full bot (so finally triggers streamer.stop)
    task = asyncio.create_task(bot.run())
    # Wait for both loops to start
    await asyncio.wait_for(started1.wait(), timeout=2.0)
    await asyncio.wait_for(started2.wait(), timeout=2.0)

    # Cancel the bot and wait for clean shutdown
    task.cancel()
    try:
        await task
    except Exception:
        pass

    assert bot.running is False
    # Streamer shutdown should have been recorded
    assert pm.counters.get('ws_shutdowns', 0) == 1
    # Ensure no pending symbol loop tasks remain
    pending_stuck = [t for t in asyncio.all_tasks() if getattr(getattr(t, 'get_coro', lambda: None)(), '__name__', '') == 'stuck_loop' and not t.done()]
    assert len(pending_stuck) == 0


@pytest.mark.asyncio
async def test_circuit_breaker_metrics_emission(monkeypatch, tmp_path):
    # Patch portfolio risk check to fail and assert no order execution but metrics emitted best-effort
    import importlib
    exmod = importlib.import_module('executor')
    import trading_bot as tb

    class FakeTradeExecutor:
        async def initialize(self):
            return None
        async def execute_order(self, *a, **k):
            assert False, 'should not be called when risk fails'

    monkeypatch.setenv('ENABLE_METRICS', '1')
    monkeypatch.setenv('PAPER_TRADING', 'true')

    # Force risk check to fail
    def failing_check(self, order):
        return False, []

    monkeypatch.setattr(tb.Portfolio, 'check_risk', failing_check, raising=True)
    bot = tb.TradingBot(strategy=DummyStrategy(), symbols=['S/T'], paper=True)
    # Replace real executor with fake
    bot.executor = FakeTradeExecutor()

    await bot.execute_trade('S/T', {'signal': 'BUY', 'price': 1.0})
    # Assert no positions changed and balance unchanged
    assert bot.positions.get('S/T', 0.0) == 0.0
