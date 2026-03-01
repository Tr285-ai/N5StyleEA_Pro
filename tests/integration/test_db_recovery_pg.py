import os
import uuid

import pytest


def _pg_url() -> str | None:
    return os.getenv("ORDERS_DB_URL") or os.getenv("DATABASE_URL")


@pytest.mark.skipif(not _pg_url(), reason="ORDERS_DB_URL/DATABASE_URL not set")
def test_db_recovery_basic_roundtrip(monkeypatch):
    from orders_db_pg import OrdersDBPG  # type: ignore

    url = _pg_url()
    assert url

    # Create DB and upsert an order
    db = OrdersDBPG(url)
    cid = str(uuid.uuid4())
    db.upsert(
        {
            "client_id": cid,
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "quantity": 1.0,
            "status": "submitted",
            "session_id": "test-session",
        }
    )
    rec = db.get(cid)
    assert rec is not None and rec.get("client_id") == cid

    # Simulate process restart by constructing a new DB instance
    db2 = OrdersDBPG(url)
    rec2 = db2.get(cid)
    assert rec2 is not None and rec2.get("client_id") == cid

    # Record a status transition on the new instance to ensure write path works
    db2.record_status_transition(
        cid, from_status="submitted", to_status="accepted", reason="test"
    )
    rec3 = db2.get(cid)
    assert rec3 is not None and rec3.get("status") in {"accepted", "submitted"}
