# Observability Stack (Prometheus, Alertmanager, Grafana, Loki, Promtail)

This folder contains a ready-to-run observability stack for local development.

## Prerequisites
- Docker and Docker Compose installed
- Port availability: 3000 (Grafana), 9090 (Prometheus), 9093 (Alertmanager), 3100 (Loki)

## Dashboards provisioning
- Grafana will auto-provision datasources and dashboards at startup.
- By default, dashboards are mounted from `ops/grafana/dashboards`.
- To override, set `GRAFANA_DASHBOARDS_PATH` before starting (absolute or relative path).

## Start the stack
- Bash (Linux/macOS/WLS):
  ```bash
  ./tools/alerts/up.sh
  ```
- PowerShell (Windows):
  ```powershell
  .\tools\alerts\up.ps1
  ```

Grafana will be available at http://localhost:3000 (default admin/admin).

## Stop the stack
- Bash:
  ```bash
  ./tools/alerts/down.sh
  ```
- PowerShell:
  ```powershell
  .\tools\alerts\down.ps1
  ```

## Services
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3000
- Loki: http://localhost:3100

## Logs
- Promtail tails `./logs` (repo root) and ships to Loki. Ensure your app writes JSON logs under `logs/`.
