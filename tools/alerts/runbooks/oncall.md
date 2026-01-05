# On-Call Runbook: Trading Observability

This runbook provides quick triage steps for Prometheus/Alertmanager alerts.

## Alerts and Playbooks

- Alert: OrdersFailedHigh5m / OrdersFailedCritical5m
  - Meaning: Order failure rate is >0 or >0.05/s over 5m.
  - Triage:
    - Check Prometheus: rate(n5_counter_orders_failed_total[5m])
    - Compare with orders received: rate(n5_counter_orders_received_total[5m])
    - Inspect executor logs around failures (order_failed events).
    - Verify exchange connectivity and credentials.
    - Check CCXT retries: rate(n5_counter_ccxt_order_retries_total[5m])
  - Mitigation:
    - If exchange unstable, consider pausing live orders (set PAPER_TRADING=true or circuit breaker)
    - Increase CCXT_BACKOFF_MS briefly; monitor.

- Alert: WsErrorsHigh5m / WsReconnectionsBurst
  - Meaning: Elevated WS errors/reconnects.
  - Triage:
    - Check rate(n5_counter_ws_errors_total[5m]) and ws_reconnects.
    - Review streamer logs for specific exception types.
    - Confirm exchange status page and network connectivity.
  - Mitigation:
    - Increase WS_RETRY_BACKOFF_MS and WS_JITTER_MS temporarily.
    - Reduce symbol load or timeframe.

- Alert: WsShutdownsDetected
  - Meaning: WS streaming shutdown occurred in last 15m.
  - Triage:
    - Confirm whether circuit breaker tripped (WS_RECONNECT_MAX).
    - Check service restarts and OOMs.
  - Mitigation:
    - Restart streamer with lower concurrency; disable chaos (WS_CHAOS_PROB=0).

- Alert: OrdersSuccessRateLow
  - Meaning: Success rate < 99% for 10m with traffic.
  - Triage:
    - Check recording rule: orders_success_rate_5m.
    - Compare failures vs received.
    - Review order parameters and risk checks.
  - Mitigation:
    - Tighten validation; temporarily halt low-confidence strategies.

- Alert: OrdersErrorBudgetBurnFast / Slow
  - Meaning: Burn-rate for 99.5% SLO is high across short/long windows.
  - Triage:
    - Review orders_error_rate_5m/30m/1h/6h.
    - Inspect recent deploys or infra events.
  - Mitigation:
    - Roll back recent changes; switch to paper.

## Useful Env Toggles

- PAPER_TRADING=true to stop placing live orders
- WS_CHAOS_PROB=0 to disable chaos
- WS_RECONNECT_MAX to bound retries
- WS_RETRY_BACKOFF_MS / WS_JITTER_MS to back off more
- ALERT_ORDERS_FAILED / ALERT_WS_ERRORS to adjust local threshold logging

## Contacts

- Exchange status pages: <add links>
- On-call rotation: <add schedule>
