import json
import os
import pytest

try:
    import jsonschema
except Exception:
    jsonschema = None


SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TCARecord",
    "type": "object",
    "required": ["mode", "event", "symbol", "timestamp"],
    "properties": {
        "mode": {"enum": ["DEMO", "LIVE"]},
        "event": {"enum": ["order", "algo_order"]},
        "symbol": {"type": "string"},
        "timestamp": {"type": "string"},
        "side": {"type": ["string", "null"]},
        "amount": {"type": ["number", "null"]},
        "arrival_price": {"type": ["number", "null"]},
        "fill_price": {"type": ["number", "null"]},
        "slippage_bps": {"type": ["number", "null"]},
        "status": {"type": ["string", "null"]},
        "order_id": {"type": ["string", "null"]},
        "child_ids": {"type": "array", "items": {"type": ["string", "null"]}},
        "child_count": {"type": ["integer", "number", "null"]},
        "session_id": {"type": ["string", "null"]},
        "exchange": {"type": ["string", "null"]},
        "algo": {"type": ["string", "null"], "enum": ["TWAP", "ICEBERG", None]},
        "latency_ms": {"type": ["object", "null"]},
    },
    "allOf": [
        {
            "if": {"properties": {"event": {"const": "algo_order"}}},
            "then": {"required": ["algo", "child_ids"]}
        }
    ]
}


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not available")
@pytest.mark.asyncio
async def test_tca_jsonl_conforms_to_schema(tmp_path, monkeypatch):
    # Use test workspace logs if present; otherwise use tmp dir
    tca_dir = 'logs'
    tca_file = os.path.join(tca_dir, 'tca.jsonl')
    if not os.path.exists(tca_file):
        pytest.skip("No TCA file to validate yet")

    with open(tca_file, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    assert len(lines) >= 1

    for ln in lines:
        rec = json.loads(ln)
        jsonschema.validate(instance=rec, schema=SCHEMA)
