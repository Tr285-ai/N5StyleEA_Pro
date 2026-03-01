# Operations Runbook

- Startup
  - Local: `docker compose -f ops/observability/docker-compose.yml up -d` (app, Prometheus, Alertmanager, Grafana)
  - App exposes: HTTP 8000, Prometheus metrics 9100
  - Set env before start (examples):
    - EXEC_ALGO=TWAP|ICEBERG|""  EXEC_ALGO_RESUME=1  EXEC_TWAP_SLICES=5  EXEC_ICEBERG_DISPLAY_PCT=0.2
    - RISK_SYMBOL_LIMITS='{"BTC/USDT":{"max_qty":1.0,"min_qty":0.001,"max_notional":50000}}'
    - RISK_MAX_QTY=1.0  RISK_ENFORCE_SLIPPAGE=0  KILL_SWITCH=0
- Health
  - Prometheus: http://localhost:9090  Grafana: http://localhost:3000 (admin/admin)
  - Key metrics: `n5_counter_*`, `n5_latency_*`, `n5_perf_error_count_total`
- Releases (CI/CD)
  - GitHub Actions runs lint, mypy, unit+integration tests with coverage ≥80%
  - Docker job builds `Dockerfile.prod` and publishes to GHCR if DOCKER_PUBLISH is true on main/tag
- SBOM & Scans
  - CI generates CycloneDX SBOM and uploads as artifact; Trivy FS scan runs on each push
- Backfill/Resume
  - TWAP/ICEBERG resume controlled by EXEC_ALGO_RESUME=1 and persisted in orders.sqlite3
  - Crash/retry is idempotent via `client_id` (OrdersDB primary key)
- Risk Controls
  - Kill switch: set KILL_SWITCH=1 to block new orders
  - Price band: `price_band_pct` blocks dev > band vs ref_price
  - Slippage: `max_slippage_bps`; enforcement via RISK_ENFORCE_SLIPPAGE=1
- Logs & TCA
  - Logs: logs/ directory; TCA JSONL: logs/tca.jsonl
  - Rotation: app attaches a TimedRotatingFileHandler writing logs/app.log daily; retention via `LOG_RETENTION_DAYS` (default 7)
  - PII: Do not log secrets, API keys, client emails/PII. Scrub sensitive fields before `log_json`. Audit access to logs via repo ACLs and CI artifacts policy.

- Database & Migrations
  - Backend: OrdersDBPG via PostgreSQL when `ORDERS_DB_BACKEND=pg` and `ORDERS_DB_URL` set
  - Migrate: `alembic -c alembic.ini upgrade head`; optional on app start with `DB_MIGRATE_ON_START=1`
  - Recovery: CI exercises reconnection/roundtrip; for manual test restart DB and verify app resumes within RTO budget

- Blue/Green Deploys (Docker)
  - Compose at ops/bluegreen/docker-compose.yml starts app_blue, app_green, and nginx (port 8080)
  - Cutover: switch upstream in ops/bluegreen/nginx.conf from app_blue to app_green and reload container
