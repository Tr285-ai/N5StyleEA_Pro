#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.observability.yml"
# Allow overriding dashboards path; default to repo path
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export REPO_ROOT
if [[ -z "${GRAFANA_DASHBOARDS_PATH:-}" ]]; then
  export GRAFANA_DASHBOARDS_PATH="$REPO_ROOT/ops/grafana/dashboards"
fi
exec docker compose -f "$COMPOSE_FILE" up -d
