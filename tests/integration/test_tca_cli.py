import io
import json
import os

import pandas as pd


def test_tca_cli_empty(tmp_path, monkeypatch, capsys):
    # Arrange
    d = tmp_path
    logs = d / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'tca.jsonl').write_text('', encoding='utf-8')
    monkeypatch.chdir(d)

    # Act
    from tools import tca_cli
    rc = tca_cli.main(['-i', 'logs/tca.jsonl', '-o', '-'])

    # Assert
    assert rc == 0
    out = capsys.readouterr().out
    # Expect header with rollup columns and no data rows
    header = 'date,symbol,exchange,trades,slippage_bps_avg,slippage_bps_p95,slippage_bps_max,slippage_cost_total'
    assert header in out
    assert len(out.strip().splitlines()) == 1


def test_tca_cli_basic(tmp_path, monkeypatch, capsys):
    # Arrange
    d = tmp_path
    logs = d / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / 'tca.jsonl'
    ts = '2024-01-01T00:00:00+00:00'
    rows = [
        {
            'timestamp': ts,
            'event': 'order',
            'symbol': 'BTC/USDT',
            'exchange': 'binance',
            'side': 'BUY',
            'amount': 1.0,
            'arrival_price': 100.0,
            'fill_price': 101.0,
        },
        {
            'timestamp': ts,
            'event': 'order',
            'symbol': 'BTC/USDT',
            'exchange': 'binance',
            'side': 'BUY',
            'amount': 1.0,
            'arrival_price': 100.0,
            'fill_price': 101.0,
        },
        {
            'timestamp': ts,
            'event': 'order',
            'symbol': 'BTC/USDT',
            'exchange': 'binance',
            'side': 'SELL',
            'amount': 2.0,
            'arrival_price': 100.0,
            'fill_price': 99.0,
        },
    ]
    with p.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.chdir(d)

    # Act
    from tools import tca_cli
    rc = tca_cli.main(['-i', 'logs/tca.jsonl', '-o', '-'])

    # Assert
    assert rc == 0
    out = capsys.readouterr().out
    df = pd.read_csv(io.StringIO(out))
    assert len(df) == 1
    rec = df.iloc[0].to_dict()
    assert rec['date'] == '2024-01-01'
    assert rec['symbol'] == 'BTC/USDT'
    assert rec['exchange'] == 'binance'
    assert int(rec['trades']) == 3
    assert abs(float(rec['slippage_bps_avg']) - 100.0) < 1e-6
    assert abs(float(rec['slippage_bps_p95']) - 100.0) < 1e-6
    assert abs(float(rec['slippage_bps_max']) - 100.0) < 1e-6
    assert abs(float(rec['slippage_cost_total']) - 3.0) < 1e-6
