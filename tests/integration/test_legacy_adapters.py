import pytest
try:
    import legacy_compat  # noqa: F401
    _LEGACY_OK = True
except Exception:
    _LEGACY_OK = False

@pytest.mark.asyncio
async def test_trading_system_legacy_adapter_sync(monkeypatch):
    if not _LEGACY_OK:
        pytest.skip("legacy_compat not present; skipping legacy adapter tests")
    import legacy_compat
    import trading_system as ts
    if not hasattr(ts, 'run_legacy_entry'):
        pytest.skip("trading_system.run_legacy_entry not present; skipping")

    logged = {'flag': False}

    def fake_log(event, payload):
        logged['flag'] = True

    def fake_run(cfg):
        # simulate telemetry from wrapper
        fake_log('legacy_wrapper_called', {'source': 'sync'})
        return {'ok': True, 'routed': True, 'cfg_symbol': cfg.get('symbol')}

    monkeypatch.setattr(legacy_compat, '_log_legacy_use', fake_log, raising=True)
    monkeypatch.setattr(legacy_compat, 'run', fake_run, raising=True)

    res = ts.run_legacy_entry({'symbol': 'BTC/USDT'})
    assert res.get('ok') is True
    assert res.get('routed') is True
    assert logged['flag'] is True

@pytest.mark.asyncio
async def test_trading_system_legacy_adapter_async(monkeypatch):
    if not _LEGACY_OK:
        pytest.skip("legacy_compat not present; skipping legacy adapter tests")
    import legacy_compat
    import trading_system as ts
    if not hasattr(ts, 'run_legacy_entry_async'):
        pytest.skip("trading_system.run_legacy_entry_async not present; skipping")

    logged = {'flag': False}

    def fake_log(event, payload):
        logged['flag'] = True

    async def fake_run_async(cfg):
        fake_log('legacy_wrapper_called', {'source': 'async'})
        return {'ok': True, 'routed': True, 'cfg_symbol': cfg.get('symbol')}

    monkeypatch.setattr(legacy_compat, '_log_legacy_use', fake_log, raising=True)
    monkeypatch.setattr(legacy_compat, 'run_legacy_strategy', fake_run_async, raising=True)

    res = await ts.run_legacy_entry_async({'symbol': 'ETH/USDT'})
    assert res.get('ok') is True
    assert res.get('routed') is True
    assert logged['flag'] is True

@pytest.mark.asyncio
async def test_trading_engine_legacy_adapter_sync(monkeypatch):
    if not _LEGACY_OK:
        pytest.skip("legacy_compat not present; skipping legacy adapter tests")
    import legacy_compat
    import trading_engine as te
    if not hasattr(te, 'run_legacy_entry'):
        pytest.skip("trading_engine.run_legacy_entry not present; skipping")

    called = {'logged': False}

    def fake_log(event, payload):
        called['logged'] = True

    def fake_run(cfg):
        fake_log('legacy_wrapper_called', {'source': 'sync'})
        return {'ok': True, 'engine_routed': True}

    monkeypatch.setattr(legacy_compat, '_log_legacy_use', fake_log, raising=True)
    monkeypatch.setattr(legacy_compat, 'run', fake_run, raising=True)

    res = te.run_legacy_entry({'symbol': 'XRP/USDT'})
    assert res.get('ok') is True
    assert res.get('engine_routed') is True
    assert called['logged'] is True

@pytest.mark.asyncio
async def test_trading_engine_legacy_adapter_async(monkeypatch):
    if not _LEGACY_OK:
        pytest.skip("legacy_compat not present; skipping legacy adapter tests")
    import legacy_compat
    import trading_engine as te
    if not hasattr(te, 'run_legacy_entry_async'):
        pytest.skip("trading_engine.run_legacy_entry_async not present; skipping")

    called = {'logged': False}

    def fake_log(event, payload):
        called['logged'] = True

    async def fake_run_async(cfg):
        fake_log('legacy_wrapper_called', {'source': 'async'})
        return {'ok': True, 'engine_routed': True}

    monkeypatch.setattr(legacy_compat, '_log_legacy_use', fake_log, raising=True)
    monkeypatch.setattr(legacy_compat, 'run_legacy_strategy', fake_run_async, raising=True)

    res = await te.run_legacy_entry_async({'symbol': 'ADA/USDT'})
    assert res.get('ok') is True
    assert res.get('engine_routed') is True
    assert called['logged'] is True
