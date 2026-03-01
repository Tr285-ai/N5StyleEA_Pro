"""
Initial orders and order_events tables

Revision ID: 0001_initial
Revises:
Create Date: 2026-02-26 00:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orders",
        sa.Column("client_id", sa.String(length=255), primary_key=True, nullable=False),
        sa.Column("symbol", sa.String(length=50), nullable=True),
        sa.Column("side", sa.String(length=16), nullable=True),
        sa.Column("type", sa.String(length=16), nullable=True),
        sa.Column("quantity", sa.Float, nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("order_id", sa.String(length=255), nullable=True),
        sa.Column("fill_price", sa.Float, nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("ts", sa.String(length=64), nullable=True),
        sa.Column("child_ids", sa.Text, nullable=True),
        sa.Column("filled_qty", sa.Float, nullable=True),
        sa.Column("remaining_qty", sa.Float, nullable=True),
        sa.Column("algo", sa.String(length=32), nullable=True),
        sa.Column("algo_total_slices", sa.BigInteger, nullable=True),
        sa.Column("algo_completed_slices", sa.BigInteger, nullable=True),
        sa.Column("algo_last_ts", sa.String(length=64), nullable=True),
    )

    op.create_table(
        "order_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("client_id", sa.String(length=255), nullable=True),
        sa.Column("from_status", sa.String(length=32), nullable=True),
        sa.Column("to_status", sa.String(length=32), nullable=True),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("ts", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("order_events")
    op.drop_table("orders")
