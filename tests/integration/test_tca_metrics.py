import os
import json
import asyncio
import pytest
import importlib

@pytest.mark.asyncio
async def test_tca_emission_demo(monkeypatch):
    monkeypatch.setenv('ENABLE_METRICS', '1')
    monkeypatch.setenv('PAPER_TRADING', 'true')

    # Ensure clean TCA file
    tca_dir = 'logs'
    tca_file = os.path.join(tca_dir, 'tca.jsonl')
    os.makedirs(tca_dir, exist_ok=True)
    try:
        if os.path.exists(tca_file):
            os.remove(tca_file)
    except Exception:
        pass

    try:
        exmod = importlib.import_module('executor')
    except Exception as e:
        pytest.skip(f"executor import failed: {e}")

    ex = exmod.TradeExecutor({'broker': {'demo': True, 'exchange': 'binance'}})
    await ex.initialize()

    # Execute one demo order to trigger TCA
    result = await ex.execute_order(
        symbol='BTC/USDT',
        side='BUY',
        order_type='MARKET',
        price=100.0,
        quantity=0.01,
    )

    assert isinstance(result, dict)

    # Verify TCA JSONL exists and contains at least one record
    assert os.path.exists(tca_file)
    with open(tca_file, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    assert len(lines) >= 1
    rec = json.loads(lines[-1])

    # Basic fields
    assert rec.get('event') == 'order'
    assert rec.get('mode') == 'DEMO'
    assert rec.get('symbol') == 'BTC/USDT'
    # slippage_bps may be None in demo; presence of key is sufficient
    assert 'slippage_bps' in rec
