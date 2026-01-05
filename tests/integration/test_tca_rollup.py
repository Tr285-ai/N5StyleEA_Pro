import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import tca_rollup as tr


def test_tca_rollup_empty(tmp_path: Path):
    p = tmp_path / 'tca.jsonl'
    p.write_text('', encoding='utf-8')
    df = tr.load_tca_jsonl(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    rolled = tr.rollup_tca(df)
    assert list(rolled.columns) == ['date','symbol','exchange','trades','slippage_bps_avg','slippage_bps_p95','slippage_bps_max','slippage_cost_total']
    assert rolled.empty


def test_tca_rollup_basic(tmp_path: Path):
    p = tmp_path / 'tca.jsonl'
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

    df = tr.load_tca_jsonl(str(p))
    assert not df.empty
    rolled = tr.rollup_tca(df)
    assert len(rolled) == 1
    rec = rolled.iloc[0].to_dict()
    assert rec['date'] == '2024-01-01'
    assert rec['symbol'] == 'BTC/USDT'
    assert rec['exchange'] == 'binance'
    assert rec['trades'] == 3
    # average/p95/max slippage should all be 100 bps across inputs
    assert abs(float(rec['slippage_bps_avg']) - 100.0) < 1e-6
    assert abs(float(rec['slippage_bps_p95']) - 100.0) < 1e-6
    assert abs(float(rec['slippage_bps_max']) - 100.0) < 1e-6
    # slippage cost: 1*1 + 1*1 + 1*2 = 3
    assert abs(float(rec['slippage_cost_total']) - 3.0) < 1e-6

    # write outputs
    cfg = tr.RollupConfig(input_path=str(p), output_csv=str(tmp_path / 'rollup.csv'), output_json=str(tmp_path / 'rollup.json'))
    tr.write_outputs(rolled, cfg)
    assert (tmp_path / 'rollup.csv').exists()
    assert (tmp_path / 'rollup.json').exists()
