import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class RollupConfig:
    input_path: str = os.path.join('logs', 'tca.jsonl')
    output_csv: Optional[str] = os.path.join('logs', 'tca_rollup.csv')
    output_json: Optional[str] = os.path.join('logs', 'tca_rollup.json')


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_date(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date().isoformat()
    except Exception:
        return "1970-01-01"


def _compute_slippage_bps(arrival: Optional[float], fill: Optional[float], side: Optional[str]) -> Optional[float]:
    if arrival is None or fill is None or arrival <= 0 or not side:
        return None
    s = str(side).lower().strip()
    try:
        if s == 'buy':
            return (fill - arrival) / arrival * 10_000.0
        if s == 'sell':
            return (arrival - fill) / arrival * 10_000.0
    except Exception:
        return None
    return None


def load_tca_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return pd.DataFrame(columns=['timestamp','symbol','exchange','side','amount','arrival_price','fill_price','slippage_bps'])
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue
    df = pd.DataFrame(rows)
    # Normalize and compute missing fields
    if 'timestamp' in df.columns:
        df['date'] = df['timestamp'].apply(_to_date)
    else:
        df['date'] = datetime.now(timezone.utc).date().isoformat()
    for col in ['arrival_price','fill_price','amount']:
        if col in df.columns:
            df[col] = df[col].apply(_safe_float)
    if 'slippage_bps' not in df.columns:
        df['slippage_bps'] = None
    # Backfill slippage_bps if missing
    def fill_slip(row):
        if row.get('slippage_bps') is not None:
            return row.get('slippage_bps')
        return _compute_slippage_bps(row.get('arrival_price'), row.get('fill_price'), row.get('side'))
    df['slippage_bps'] = df.apply(fill_slip, axis=1)
    # Filter only order events with symbol
    if 'event' in df.columns:
        df = df[df['event'].isin(['order','algo_order'])]
    if 'symbol' in df.columns:
        df = df[df['symbol'].notna()]
    else:
        # No symbol column => nothing to roll up
        return pd.DataFrame(columns=df.columns)
    return df


def rollup_tca(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['date','symbol','exchange','trades','slippage_bps_avg','slippage_bps_p95','slippage_bps_max','slippage_cost_total'])
    # compute slippage cost as absolute slippage percent per trade (sum of |bps|/100)
    def slip_cost(row) -> float:
        try:
            bps = float(row.get('slippage_bps'))
        except Exception:
            bps = 0.0
        return abs(bps) / 100.0
    df = df.copy()
    df['slippage_cost'] = df.apply(slip_cost, axis=1)
    # ensure exchange column
    if 'exchange' not in df.columns:
        df['exchange'] = 'unknown'
    grp = df.groupby(['date','symbol','exchange'])
    out = grp.agg(
        trades=('symbol','count'),
        slippage_bps_avg=('slippage_bps', lambda x: float(pd.Series(x, dtype='float64').dropna().mean()) if len(pd.Series(x).dropna())>0 else 0.0),
        slippage_bps_p95=('slippage_bps', lambda x: float(pd.Series(x, dtype='float64').dropna().quantile(0.95)) if len(pd.Series(x).dropna())>0 else 0.0),
        slippage_bps_max=('slippage_bps', lambda x: float(pd.Series(x, dtype='float64').dropna().max()) if len(pd.Series(x).dropna())>0 else 0.0),
        slippage_cost_total=('slippage_cost','sum'),
    ).reset_index()
    return out


def write_outputs(df: pd.DataFrame, cfg: RollupConfig) -> None:
    os.makedirs(os.path.dirname(cfg.output_csv or cfg.input_path), exist_ok=True)
    if cfg.output_csv:
        df.to_csv(cfg.output_csv, index=False)
    if cfg.output_json:
        with open(cfg.output_json, 'w', encoding='utf-8') as f:
            json.dump(json.loads(df.to_json(orient='records')), f, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = RollupConfig()
    df = load_tca_jsonl(cfg.input_path)
    rolled = rollup_tca(df)
    write_outputs(rolled, cfg)


if __name__ == '__main__':
    main()
