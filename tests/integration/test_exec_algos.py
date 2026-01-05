import os
import json
import pytest
import importlib

@pytest.mark.asyncio
async def test_twap_algo_emits_tca(monkeypatch):
    # Ensure metrics and TWAP settings
    monkeypatch.setenv('ENABLE_METRICS', '1')
    monkeypatch.setenv('EXEC_ALGO', 'TWAP')
    monkeypatch.setenv('EXEC_TWAP_SLICES', '3')
    monkeypatch.setenv('EXEC_TWAP_DURATION_SEC', '0')  # fast

    # Prepare TCA file
    tca_dir = 'logs'
    tca_file = os.path.join(tca_dir, 'tca.jsonl')
    os.makedirs(tca_dir, exist_ok=True)
    try:
        if os.path.exists(tca_file):
            os.remove(tca_file)
    except Exception:
        pass

    # Import executor module safely
    try:
        exmod = importlib.import_module('executor')
    except Exception as e:
        pytest.skip(f"executor import failed: {e}")

    # Fake executor to avoid ccxt dependency (constant price for TWAP)
    class FakeExec:
        def __init__(self, exchange_name: str, api_key: str, api_secret: str, paper: bool):
            self.exchange_name = exchange_name
            self.paper = False
        async def place_market_order(self, symbol: str, side: str, amount: float, params=None):
            return {
                'id': f'FAKE_{symbol}_{side}_{amount}',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'status': 'filled',
                'average': 101.0,
            }
        async def get_ticker(self, symbol: str):
            return {'symbol': symbol, 'last': 100.0}

    # Patch executor and sleep
    monkeypatch.setattr(exmod, 'Executor', FakeExec, raising=True)
    async def fast_sleep(_):
        return None
    monkeypatch.setattr(exmod.asyncio, 'sleep', fast_sleep, raising=True)

    # Create trade executor in live mode
    te = exmod.TradeExecutor({'broker': {'demo': False, 'exchange': 'binance', 'api_key': 'k', 'api_secret': 's'}})
    await te.initialize()

    # Execute order (TWAP)
    res = await te.execute_order(symbol='BTC/USDT', side='BUY', order_type='MARKET', price=None, quantity=1.0)
    assert isinstance(res, dict)
    assert res.get('status') in ('filled', 'partial')

    # Verify TCA record
    assert os.path.exists(tca_file)
    with open(tca_file, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert rec.get('event') == 'algo_order'
    assert rec.get('algo') == 'TWAP'
    assert rec.get('mode') == 'LIVE'
    # TCA content checks
    assert isinstance(rec.get('child_ids'), list)
    assert len(rec['child_ids']) == 3
    assert rec.get('fill_price') == pytest.approx(101.0, rel=1e-6)
    assert rec.get('arrival_price') == pytest.approx(100.0, rel=1e-6)
    assert rec.get('slippage_bps') is None or rec.get('slippage_bps') >= 0


@pytest.mark.asyncio
async def test_iceberg_algo_emits_tca(monkeypatch):
    # Ensure metrics and ICEBERG settings
    monkeypatch.setenv('ENABLE_METRICS', '1')
    monkeypatch.setenv('EXEC_ALGO', 'ICEBERG')
    # Use 40% display to produce 3 chunks (0.4, 0.4, 0.2 of total)
    monkeypatch.setenv('EXEC_ICEBERG_DISPLAY_PCT', '0.4')

    # Prepare TCA file
    tca_dir = 'logs'
    tca_file = os.path.join(tca_dir, 'tca.jsonl')
    os.makedirs(tca_dir, exist_ok=True)
    try:
        if os.path.exists(tca_file):
            os.remove(tca_file)
    except Exception:
        pass

    # Import executor module safely
    try:
        exmod = importlib.import_module('executor')
    except Exception as e:
        pytest.skip(f"executor import failed: {e}")

    # Fake executor to avoid ccxt dependency (variable prices for weighted avg)
    class FakeExec:
        def __init__(self, exchange_name: str, api_key: str, api_secret: str, paper: bool):
            self.exchange_name = exchange_name
            self.paper = False
            self.calls = 0
            self.amounts = []
            self.prices = []
        async def place_market_order(self, symbol: str, side: str, amount: float, params=None):
            self.calls += 1
            avg_price = 100.0 if self.calls == 1 else 110.0 if self.calls == 2 else 120.0
            self.amounts.append(float(amount))
            self.prices.append(float(avg_price))
            return {
                'id': f'FAKE_{symbol}_{side}_{amount}',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'status': 'filled',
                'average': avg_price,
            }
        async def get_ticker(self, symbol: str):
            return {'symbol': symbol, 'last': 100.0}

    # Patch executor and sleep
    monkeypatch.setattr(exmod, 'Executor', FakeExec, raising=True)
    async def fast_sleep(_):
        return None
    monkeypatch.setattr(exmod.asyncio, 'sleep', fast_sleep, raising=True)

    # Create trade executor in live mode
    te = exmod.TradeExecutor({'broker': {'demo': False, 'exchange': 'binance', 'api_key': 'k', 'api_secret': 's'}})
    await te.initialize()

    # Execute order (ICEBERG)
    res = await te.execute_order(symbol='ETH/USDT', side='BUY', order_type='MARKET', price=None, quantity=1.0)
    assert isinstance(res, dict)
    assert res.get('status') in ('filled', 'partial')

    # Verify TCA record
    assert os.path.exists(tca_file)
    with open(tca_file, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert rec.get('event') == 'algo_order'
    assert rec.get('algo') == 'ICEBERG'
    assert rec.get('mode') == 'LIVE'
    # TCA content checks
    assert isinstance(rec.get('child_ids'), list)
    # Expect 3 chunks with display pct 0.4 for total amount 1.0: amounts 0.4, 0.4, 0.2
    assert len(rec['child_ids']) == 3
    # Compute expected weighted average from the actual child amounts/prices recorded by the fake executor
    amounts = te._exchange_executor.amounts  # type: ignore[attr-defined]
    prices = te._exchange_executor.prices  # type: ignore[attr-defined]
    denom = sum(amounts)
    expected_wavg = sum(a*p for a, p in zip(amounts, prices)) / denom if denom else 0.0
    assert rec.get('fill_price') == pytest.approx(expected_wavg, rel=1e-6)
    assert rec.get('arrival_price') == pytest.approx(100.0, rel=1e-6)
    assert rec.get('slippage_bps') is None or rec.get('slippage_bps') >= 0
