import os
import json
import importlib
import pytest


@pytest.mark.asyncio
async def test_twap_partial_fill_emits_partial_status(monkeypatch, tmp_path):
    monkeypatch.setenv('ENABLE_METRICS', '1')
    monkeypatch.setenv('EXEC_ALGO', 'TWAP')
    monkeypatch.setenv('EXEC_TWAP_SLICES', '2')
    monkeypatch.setenv('EXEC_TWAP_DURATION_SEC', '0')

    # Prepare TCA file
    tca_dir = tmp_path / 'logs'
    tca_dir.mkdir(parents=True, exist_ok=True)
    tca_file = tca_dir / 'tca.jsonl'
    monkeypatch.chdir(tmp_path)

    exmod = importlib.import_module('executor')

    class FakeExec:
        def __init__(self, exchange_name: str, api_key: str, api_secret: str, paper: bool):
            self.exchange_name = exchange_name
            self.paper = False
            self.calls = 0
        async def place_market_order(self, symbol: str, side: str, amount: float, params=None):
            self.calls += 1
            # First child fills, second child raises to simulate failure/partial
            if self.calls == 1:
                return {
                    'id': f'FAKE_{symbol}_{side}_{amount}',
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'status': 'filled',
                    'average': 101.0,
                }
            raise RuntimeError('transient')
        async def get_ticker(self, symbol: str):
            return {'symbol': symbol, 'last': 100.0}

    # Patch executor and sleep
    monkeypatch.setattr(exmod, 'Executor', FakeExec, raising=True)
    async def fast_sleep(_):
        return None
    monkeypatch.setattr(exmod.asyncio, 'sleep', fast_sleep, raising=True)

    te = exmod.TradeExecutor({'broker': {'demo': False, 'exchange': 'binance', 'api_key': 'k', 'api_secret': 's'}})
    await te.initialize()

    # Execute order (TWAP)
    out = await te.execute_order(symbol='BTC/USDT', side='BUY', order_type='MARKET', price=None, quantity=1.0)

    assert isinstance(out, dict)
    assert out.get('status') == 'partial'

    # Verify TCA record emitted with partial status
    assert tca_file.exists()
    lines = tca_file.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert rec.get('event') == 'algo_order'
    assert rec.get('algo') == 'TWAP'
    assert rec.get('status') == 'partial'
