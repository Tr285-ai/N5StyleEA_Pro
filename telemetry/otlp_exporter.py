"""
Minimal OpenTelemetry OTLP exporter hook for performance_monitor counters.
- If OpenTelemetry packages are not installed, setup_metrics_exporter returns a no-op hook.
- If installed, it configures a MeterProvider with an OTLP HTTP exporter and returns
  a hook that maps counter names to OTel Counter instruments and records increments.
Configuration via environment variables (all optional):
- OTLP_ENDPOINT (default: http://localhost:4318)
- OTLP_HEADERS (comma-separated key=value)
- OTLP_SERVICE_NAME (default: n5styleea)
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import os


def _parse_headers(hs: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in hs.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def setup_metrics_exporter() -> Optional[Callable[[str, int], None]]:
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    except Exception:
        # OpenTelemetry not available; return a no-op hook
        def noop(name: str, value: int) -> None:
            return None
        return noop

    endpoint = os.getenv('OTLP_ENDPOINT', 'http://localhost:4318')
    headers = _parse_headers(os.getenv('OTLP_HEADERS', ''))
    service_name = os.getenv('OTLP_SERVICE_NAME', 'n5styleea')

    resource = Resource.create({SERVICE_NAME: service_name})
    exporter = OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics", headers=headers or None)
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter(service_name)

    instruments: Dict[str, any] = {}

    def hook(counter_name: str, inc: int) -> None:
        try:
            inst = instruments.get(counter_name)
            if inst is None:
                # Prom-style suffix _total is common; keep raw name for clarity
                inst = meter.create_counter(name=f"n5_{counter_name}", description=f"Counter for {counter_name}")
                instruments[counter_name] = inst
            inst.add(int(inc))
        except Exception:
            # Never fail caller on telemetry errors
            pass

    return hook
