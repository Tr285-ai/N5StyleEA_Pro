from __future__ import annotations
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

try:
    from orders_db_pg import Base  # type: ignore
except Exception:
    Base = None  # type: ignore

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = getattr(Base, "metadata", None)


def _get_url() -> str | None:
    return os.getenv("ORDERS_DB_URL") or os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    url = _get_url()
    if not url:
        raise RuntimeError("Database URL not set. Provide ORDERS_DB_URL or DATABASE_URL or set sqlalchemy.url in alembic.ini")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _get_url()
    if not url:
        raise RuntimeError("Database URL not set. Provide ORDERS_DB_URL or DATABASE_URL or set sqlalchemy.url in alembic.ini")
    connectable = engine_from_config(
        {"sqlalchemy.url": url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
