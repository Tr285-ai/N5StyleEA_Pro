#requires -Version 5.1
$ErrorActionPreference = 'Stop'

# Determine paths
$composeFile = Join-Path $PSScriptRoot 'docker-compose.observability.yml'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

# Set environment variables for docker compose
if (-not $env:GRAFANA_DASHBOARDS_PATH -or $env:GRAFANA_DASHBOARDS_PATH -eq '') {
    $env:GRAFANA_DASHBOARDS_PATH = Join-Path $repoRoot 'ops\grafana\dashboards'
}
# Use forward slashes for docker path interpolation
$env:REPO_ROOT = ($repoRoot -replace '\\','/')

Write-Host "GRAFANA_DASHBOARDS_PATH=$($env:GRAFANA_DASHBOARDS_PATH)"
Write-Host "REPO_ROOT=$($env:REPO_ROOT)"

# Start the stack
& docker compose -f $composeFile up -d
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed with exit code $LASTEXITCODE" }
Write-Host 'Observability stack started. Grafana: http://localhost:3000  Prometheus: http://localhost:9090  Alertmanager: http://localhost:9093  Loki: http://localhost:3100'
