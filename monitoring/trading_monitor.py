""
Trading System Monitoring and Alerting

This module provides real-time monitoring, alerting, and performance metrics
for the trading infrastructure.
"""
import time
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import threading
import queue
import psutil
import socket
import os
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """Represents a single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }

@dataclass
class Alert:
    """Represents an alert condition and its state."""
    name: str
    condition: Callable[[Dict[str, float]], bool]
    severity: str  # 'info', 'warning', 'critical'
    message: str
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, metrics: Dict[str, float]) -> bool:
        """Check if the alert condition is met."""
        try:
            if self.condition(metrics):
                now = datetime.utcnow()
                if (self.last_triggered is None or 
                    (now - self.last_triggered).total_seconds() >= self.cooldown):
                    self.last_triggered = now
                    return True
            return False
        except Exception as e:
            logger.error(f"Error evaluating alert {self.name}: {e}")
            return False

class TradingMonitor:
    """Monitors the trading system and triggers alerts."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the trading monitor."""
        self.config = config or {}
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: List[Alert] = []
        self.alert_handlers = []
        self.metric_handlers = []
        self.running = False
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metric_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        
        # System information
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        
        # Register default alerts
        self._register_default_alerts()
    
    def _register_default_alerts(self):
        """Register default alert conditions."""
        # High latency alert
        self.add_alert(
            name="high_order_latency",
            condition=lambda m: m.get('order_processing_latency_ms', 0) > 100,  # 100ms
            severity="warning",
            message="High order processing latency detected"
        )
        
        # High CPU usage alert
        self.add_alert(
            name="high_cpu_usage",
            condition=lambda m: m.get('system.cpu.percent', 0) > 90,  # 90%
            severity="warning",
            message="High CPU usage detected"
        )
        
        # Network connectivity alert
        self.add_alert(
            name="network_connectivity",
            condition=lambda m: m.get('network.ping.exchange', 0) > 1000,  # 1000ms
            severity="critical",
            message="High latency to exchange detected"
        )
    
    def start(self):
        """Start the monitoring service."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("Trading monitor started")
    
    def stop(self):
        """Stop the monitoring service."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.executor.shutdown(wait=True)
    
    def _run(self):
        """Main monitoring loop."""
        last_metrics_update = 0
        last_system_update = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Process incoming metrics
                self._process_metric_queue()
                
                # Update system metrics every 5 seconds
                if current_time - last_system_update >= 5:
                    self._update_system_metrics()
                    last_system_update = current_time
                
                # Process alerts every second
                if current_time - last_metrics_update >= 1:
                    self._check_alerts()
                    last_metrics_update = current_time
                
                # Process alert queue
                self._process_alert_queue()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on error
    
    def _process_metric_queue(self):
        """Process metrics from the queue."""
        try:
            while True:
                metric = self.metric_queue.get_nowait()
                self._record_metric(metric)
        except queue.Empty:
            pass
    
    def _process_alert_queue(self):
        """Process alerts from the queue."""
        try:
            while True:
                alert = self.alert_queue.get_nowait()
                self._handle_alert(alert)
        except queue.Empty:
            pass
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric('system.cpu.percent', cpu_percent)
            
            # Memory usage
            mem = psutil.virtual_memory()
            self.record_metric('system.memory.percent', mem.percent)
            self.record_metric('system.memory.used_gb', mem.used / (1024**3))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric('system.disk.percent', disk.percent)
            self.record_metric('system.disk.used_gb', disk.used / (1024**3))
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.record_metric('network.bytes_sent', net_io.bytes_sent)
            self.record_metric('network.bytes_recv', net_io.bytes_recv)
            
            # Process metrics
            process = psutil.Process()
            self.record_metric('process.cpu.percent', process.cpu_percent(interval=None))
            self.record_metric('process.memory.rss_gb', process.memory_info().rss / (1024**3))
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _record_metric(self, metric: Metric):
        """Record a metric."""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        
        # Keep only the last N metrics
        max_metrics = self.config.get('max_metrics_per_series', 1000)
        if len(self.metrics[metric.name]) >= max_metrics:
            self.metrics[metric.name].pop(0)
        
        self.metrics[metric.name].append(metric)
        
        # Notify metric handlers
        for handler in self.metric_handlers:
            try:
                handler(metric)
            except Exception as e:
                logger.error(f"Error in metric handler: {e}")
    
    def _check_alerts(self):
        """Check all registered alerts."""
        if not self.alerts:
            return
        
        # Get current metric values
        current_metrics = {}
        for name, metrics in self.metrics.items():
            if metrics:
                current_metrics[name] = metrics[-1].value
        
        # Check each alert
        for alert in self.alerts:
            if alert.should_trigger(current_metrics):
                self.alert_queue.put({
                    'name': alert.name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': datetime.utcnow().isoformat(),
                    'metrics': current_metrics
                })
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle an alert."""
        logger.warning(f"ALERT [{alert['severity'].upper()}] {alert['name']}: {alert['message']}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a new metric."""
        metric = Metric(
            name=name,
            value=float(value),
            tags=tags or {}
        )
        self.metric_queue.put(metric)
    
    def add_alert(self, name: str, condition: Callable, severity: str, message: str, cooldown: int = 300):
        """Add a new alert condition."""
        alert = Alert(
            name=name,
            condition=condition,
            severity=severity,
            message=message,
            cooldown=cooldown
        )
        self.alerts.append(alert)
    
    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register a callback for alerts."""
        self.alert_handlers.append(handler)
    
    def register_metric_handler(self, handler: Callable[[Metric], None]):
        """Register a callback for new metrics."""
        self.metric_handlers.append(handler)
    
    def get_metrics(self, name: str, limit: int = 100) -> List[Metric]:
        """Get recent metrics by name."""
        return self.metrics.get(name, [])[-limit:]
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        return list(self.metrics.keys())

# Example alert handlers
def log_alert(alert: Dict[str, Any]):
    """Log alerts to a file."""
    with open('alerts.log', 'a') as f:
        f.write(f"{alert['timestamp']} [{alert['severity']}] {alert['name']}: {alert['message']}\n")

def send_slack_alert(webhook_url: str):
    """Create a handler that sends alerts to Slack."""
    import requests
    
    def handler(alert: Dict[str, Any]):
        try:
            message = {
                'text': f"*[{alert['severity'].upper()}] {alert['name']}*\n{alert['message']}",
                'username': 'Trading System Monitor',
                'icon_emoji': ':warning:' if alert['severity'] == 'warning' else ':exclamation:'
            }
            requests.post(webhook_url, json=message)
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    return handler

# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = TradingMonitor()
    
    # Register alert handlers
    monitor.register_alert_handler(log_alert)
    
    # Start monitoring
    monitor.start()
    
    try:
        # Simulate some metrics
        import random
        
        while True:
            # Simulate order processing latency
            latency = random.uniform(10, 200)  # 10-200ms
            monitor.record_metric('order_processing_latency_ms', latency)
            
            # Simulate exchange ping
            ping = random.uniform(5, 150)  # 5-150ms
            monitor.record_metric('network.ping.exchange', ping)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop()
