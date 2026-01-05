import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import psutil
import os
from dataclasses import dataclass, asdict
import json
from logging_json import log_json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_mb: float
    execution_time_ms: float
    latency_ms: Dict[str, float]
    throughput: Dict[str, float]
    error_count: int = 0
    warning_count: int = 0
    custom_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class PerformanceMonitor:
    """
    Monitors and logs system and trading performance metrics.
    """
    def __init__(self, log_interval: int = 60):
        """
        Initialize the performance monitor.
        
        Args:
            log_interval: Log metrics every N seconds
        """
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.metrics_history = []
        self._start_times = {}
        self._latency_metrics = {}
        self._throughput_counters = {}
        self._error_count = 0
        self._warning_count = 0
        self._thresholds: Dict[str, int] = self._load_thresholds()
        self._alerted: set[str] = set()
        self._counter_hook: Optional[Callable[[str, int], None]] = None
        
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._start_times[name] = time.time()
        
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time in milliseconds."""
        if name not in self._start_times:
            return 0.0
            
        elapsed_ms = (time.time() - self._start_times[name]) * 1000
        
        # Update latency metrics
        if name not in self._latency_metrics:
            self._latency_metrics[name] = []
        self._latency_metrics[name].append(elapsed_ms)
        
        # Keep only last 1000 samples
        if len(self._latency_metrics[name]) > 1000:
            self._latency_metrics[name] = self._latency_metrics[name][-1000:]
            
        return elapsed_ms
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a throughput counter."""
        if name not in self._throughput_counters:
            self._throughput_counters[name] = 0
        self._throughput_counters[name] += value
        # Threshold checks for selected metrics
        self._check_threshold(name)
        # Optional external hook (e.g., OTLP exporter)
        try:
            if self._counter_hook is not None:
                self._counter_hook(name, int(value))
        except Exception:
            pass
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self._error_count += 1
    
    def record_warning(self) -> None:
        """Record a warning occurrence."""
        self._warning_count += 1

    def export_counters(self) -> Dict[str, float]:
        return dict(self._throughput_counters)

    def export_error_count(self) -> int:
        return int(self._error_count)

    def export_warning_count(self) -> int:
        return int(self._warning_count)

    def reset_counters(self) -> None:
        self._throughput_counters = {}
        self._error_count = 0
        self._warning_count = 0
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        process = psutil.Process(os.getpid())
        
        # Calculate latency statistics
        latency_stats = {}
        for name, values in self._latency_metrics.items():
            if values:
                latency_stats[f"{name}_avg_ms"] = float(np.mean(values))
                latency_stats[f"{name}_p95_ms"] = float(np.percentile(values, 95))
                latency_stats[f"{name}_max_ms"] = float(max(values))
        
        # Calculate throughput rates
        throughput_stats = {}
        current_time = time.time()
        time_elapsed = current_time - self.last_log_time if self.last_log_time > 0 else 1.0
        
        for name, count in self._throughput_counters.items():
            throughput_stats[f"{name}_per_second"] = count / time_elapsed
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / (1024 * 1024),  # Convert to MB
            execution_time_ms=time.time() - self._start_times.get('total', current_time) * 1000,
            latency_ms=latency_stats,
            throughput=throughput_stats,
            error_count=self._error_count,
            warning_count=self._warning_count
        )
        
        # Reset counters
        self._throughput_counters = {}
        self._error_count = 0
        self._warning_count = 0
        self.last_log_time = current_time
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 records
            self.metrics_history = self.metrics_history[-1000:]
            
        return metrics
    
    def log_metrics(self, level: int = logging.INFO) -> None:
        """Log current metrics at specified log level."""
        metrics = self.collect_metrics()
        logger.log(level, f"Performance Metrics: {metrics.to_json()}")
    
    def get_metric_history(self, metric_name: str, window: Optional[timedelta] = None) -> List[Tuple[datetime, float]]:
        """Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric (e.g., 'cpu_percent', 'latency_ms.signal_generation_avg_ms')
            window: Optional time window to retrieve metrics for
            
        Returns:
            List of (timestamp, value) tuples
        """
        result = []
        now = datetime.now(timezone.utc)
        
        for metrics in self.metrics_history:
            timestamp = datetime.fromisoformat(metrics.timestamp)
            if window and (now - timestamp) > window:
                continue
                
            # Handle nested metrics (e.g., latency_ms.signal_generation_avg_ms)
            value = metrics
            for part in metric_name.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
                    
            if value is not None:
                result.append((timestamp, value))
                
        return result

    def set_counter_hook(self, hook: Optional[Callable[[str, int], None]]) -> None:
        """Register an optional hook to receive counter increments (e.g., OTLP export)."""
        self._counter_hook = hook

    # --- Lightweight Prometheus HTTP server (optional) ---
    def _make_http_handler(self):
        monitor = self
        from http.server import BaseHTTPRequestHandler
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == '/metrics':
                        body = monitor.render_prometheus().encode('utf-8')
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
                        self.send_header('Content-Length', str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception:
                    try:
                        self.send_response(500)
                        self.end_headers()
                    except Exception:
                        pass
            def log_message(self, format, *args):  # noqa: N802 (fastapi-style)
                try:
                    logger.debug("prom_http: " + (format % args))
                except Exception:
                    pass
        return _Handler

    def start_prometheus_http_server(self, host: str = '0.0.0.0', port: int = 0) -> Optional[int]:
        """Start a tiny HTTP server to expose /metrics. Returns bound port or None on failure."""
        if port <= 0:
            return None
        try:
            import socketserver
            import threading
            Handler = self._make_http_handler()
            httpd = socketserver.TCPServer((host, port), Handler)
            bound_port = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            logger.info(f"Prometheus metrics server started on {host}:{bound_port}")
            return bound_port
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            return None

    def _load_thresholds(self) -> Dict[str, int]:
        """Load alert thresholds from environment variables."""
        thresholds: Dict[str, int] = {}
        try:
            val = int(os.getenv('ALERT_ORDERS_FAILED', '0'))
            if val > 0:
                thresholds['orders_failed'] = val
        except Exception:
            pass
        try:
            val = int(os.getenv('ALERT_WS_ERRORS', '0'))
            if val > 0:
                thresholds['ws_errors'] = val
        except Exception:
            pass
        return thresholds

    def set_threshold(self, metric: str, value: int) -> None:
        if value > 0:
            self._thresholds[metric] = int(value)
        elif metric in self._thresholds:
            del self._thresholds[metric]
        # allow re-alerting if reset
        if metric in self._alerted:
            self._alerted.remove(metric)

    def _check_threshold(self, name: str) -> None:
        th = self._thresholds.get(name)
        if th is None:
            return
        val = int(self._throughput_counters.get(name, 0))
        if val >= th and name not in self._alerted:
            try:
                log_json('alert_threshold', metric=name, value=val, threshold=th)
            except Exception:
                pass
            self._alerted.add(name)

    def render_prometheus(self) -> str:
        """Render current counters and error/warning counts in Prometheus exposition format."""
        lines: List[str] = []
        counters = self.export_counters()
        # Generic counters
        for k, v in sorted(counters.items()):
            metric = f"n5_counter_{k}_total"
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"# HELP {metric} Arbitrary counter {k}")
            lines.append(f"{metric} {int(v)}")
        # Error/warning counts
        lines.append("# TYPE n5_perf_error_count_total counter")
        lines.append("# HELP n5_perf_error_count_total Error count since last render")
        lines.append(f"n5_perf_error_count_total {self.export_error_count()}")
        lines.append("# TYPE n5_perf_warning_count_total counter")
        lines.append("# HELP n5_perf_warning_count_total Warning count since last render")
        lines.append(f"n5_perf_warning_count_total {self.export_warning_count()}")
        return "\n".join(lines) + "\n"

# Singleton instance
performance_monitor = PerformanceMonitor()

# Auto-start Prometheus metrics HTTP server if configured via env
try:
    _prom_port = int(os.getenv('PROM_SERVER_PORT', '0'))
    if _prom_port > 0:
        _prom_host = os.getenv('PROM_SERVER_HOST', '0.0.0.0')
        performance_monitor.start_prometheus_http_server(_prom_host, _prom_port)
except Exception:
    pass

# Auto-setup OTLP metrics exporter if enabled
try:
    if os.getenv('OTLP_ENABLE', '').lower() in {'1','true','yes','on'}:
        from telemetry.otlp_exporter import setup_metrics_exporter
        hook = setup_metrics_exporter()
        if hook is not None:
            performance_monitor.set_counter_hook(hook)
except Exception:
    pass

def monitor_performance(interval: int = 60):
    """Decorator to monitor function execution time and errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = performance_monitor
            func_name = func.__name__
            
            # Start timer
            monitor.start_timer(func_name)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                execution_time = monitor.stop_timer(func_name)
                monitor.increment_counter(f"{func_name}_calls")
                
                # Log if needed
                if time.time() - monitor.last_log_time > monitor.log_interval:
                    monitor.log_metrics()
                    
                return result
                
            except Exception as e:
                # Record error
                monitor.record_error()
                monitor.stop_timer(func_name)
                monitor.increment_counter(f"{func_name}_errors")
                logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator
