import json
import logging
import re

import pytest

from logging_json import log_json
from performance_monitor import performance_monitor


@pytest.mark.parametrize(
    "event,fields",
    [
        ("ws_reconnect", {"exchange": "binance", "symbol": "BTC/USDT", "attempt": 1}),
        ("order_failed", {"symbol": "ETH/USDT", "side": "BUY", "error": "simulated"}),
    ],
)
def test_log_json_snapshot(caplog, event, fields):
    with caplog.at_level(logging.INFO):
        log_json(event, **fields)
    assert len(caplog.records) >= 1
    rec = caplog.records[-1]
    payload = json.loads(rec.getMessage())

    # Stable schema assertions (snapshot-style without brittle values)
    assert payload.get("event") == event
    assert "timestamp" in payload and isinstance(payload["timestamp"], str)
    # Ensure ISO8601 with timezone info
    assert re.search(r"\+00:00$", payload["timestamp"]) is not None

    # Required fields propagate
    for k, v in fields.items():
        assert payload.get(k) == v


def test_alert_threshold_log_emitted(caplog):
    # Force threshold emission for a known counter
    performance_monitor.set_threshold("orders_failed", 1)
    with caplog.at_level(logging.INFO):
        performance_monitor.increment_counter("orders_failed", 1)
    # Find the alert_threshold record
    matches = []
    for rec in caplog.records:
        try:
            obj = json.loads(rec.getMessage())
        except Exception:
            continue
        if obj.get("event") == "alert_threshold":
            matches.append(obj)
    assert len(matches) >= 1
    last = matches[-1]
    assert last.get("metric") == "orders_failed"
    assert int(last.get("value", 0)) >= int(last.get("threshold", 1))
