#requires -Version 5.1
$ErrorActionPreference = 'Stop'

$composeFile = Join-Path $PSScriptRoot 'docker-compose.observability.yml'

& docker compose -f $composeFile down
if ($LASTEXITCODE -ne 0) { throw "docker compose down failed with exit code $LASTEXITCODE" }
Write-Host 'Observability stack stopped.'
