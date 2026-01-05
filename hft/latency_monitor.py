import time
import numpy as np
from numba import njit
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading
import statistics
import psutil
import os
from collections import defaultdict, deque
import json

@dataclass
class LatencyStats:
    """Container for latency statistics."""
    count: int = 0
    total_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0
    sum_squares: float = 0.0
    percentiles: Dict[float, float] = field(default_factory=dict)
    
    def update(self, latency_ns: int):
        """Update statistics with a new latency measurement."""
        self.count += 1
        self.total_ns += latency_ns
        self.sum_squares += (latency_ns ** 2)
        
        if self.count == 1:
            self.min_ns = self.max_ns = latency_ns
        else:
            self.min_ns = min(self.min_ns, latency_ns)
            self.max_ns = max(self.max_ns, latency_ns)
    
    def calculate_stats(self) -> Dict[str, float]:
        """Calculate and return statistics."""
        if self.count == 0:
            return {}
            
        avg = self.total_ns / self.count
        variance = (self.sum_squares / self.count) - (avg ** 2)
        std_dev = np.sqrt(variance) if variance > 0 else 0
        
        return {
            'count': self.count,
            'avg_ns': avg,
            'min_ns': self.min_ns,
            'max_ns': self.max_ns,
            'std_dev_ns': std_dev,
            'p50_ns': self.percentiles.get(50, 0),
            'p95_ns': self.percentiles.get(95, 0),
            'p99_ns': self.percentiles.get(99, 0),
            'p999_ns': self.percentiles.get(99.9, 0)
        }

class LatencyMonitor:
    """High-performance latency monitoring for HFT systems."""
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.metrics: Dict[str, List[int]] = defaultdict(lambda: deque(maxlen=window_size))
        self.stats: Dict[str, LatencyStats] = defaultdict(LatencyStats)
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.last_update = time.time()
        self.start_time = time.time()
        
        # System metrics
        self.cpu_percent = 0.0
        self.memory_mb = 0.0
        self.net_io = (0, 0)  # (bytes_sent, bytes_recv)
        
        # Start background thread
        self.start()
    
    def start(self):
        """Start the monitoring thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def record(self, metric_name: str, latency_ns: int):
        """Record a latency measurement."""
        with self.lock:
            self.metrics[metric_name].append(latency_ns)
            self.stats[metric_name].update(latency_ns)
    
    def _monitor_loop(self):
        """Background thread for periodic monitoring."""
        last_net_io = psutil.net_io_counters()
        
        while self.running:
            try:
                # Update system metrics
                self.cpu_percent = psutil.cpu_percent(interval=0.1)
                self.memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                
                # Update network I/O
                net_io = psutil.net_io_counters()
                self.net_io = (
                    net_io.bytes_sent - last_net_io.bytes_sent,
                    net_io.bytes_recv - last_net_io.bytes_recv
                )
                last_net_io = net_io
                
                # Calculate percentiles periodically
                self._calculate_percentiles()
                
                # Sleep for a short interval (100ms)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def _calculate_percentiles(self):
        """Calculate percentiles for all metrics."""
        with self.lock:
            for metric, values in self.metrics.items():
                if values:
                    # Convert deque to numpy array for efficient calculation
                    arr = np.array(values, dtype=np.int64)
                    
                    # Calculate percentiles
                    self.stats[metric].percentiles = {
                        50: float(np.percentile(arr, 50)),
                        90: float(np.percentile(arr, 90)),
                        95: float(np.percentile(arr, 95)),
                        99: float(np.percentile(arr, 99)),
                        99.9: float(np.percentile(arr, 99.9))
                    }
                    
                    # Clear the values to maintain the sliding window
                    self.metrics[metric].clear()
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics with statistics."""
        result = {}
        with self.lock:
            for name, stats in self.stats.items():
                result[name] = stats.calculate_stats()
        return result
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system-level metrics."""
        return {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'network_sent_bytes': self.net_io[0],
            'network_recv_bytes': self.net_io[1]
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of metrics."""
        metrics = self.get_metrics()
        system = self.get_system_metrics()
        
        summary = ["=== Latency Metrics ==="]
        for name, stats in metrics.items():
            if stats['count'] > 0:
                summary.append(
                    f"{name}: "
                    f"avg={stats['avg_ns']/1000:.2f}µs "
                    f"min={stats['min_ns']/1000:.2f}µs "
                    f"max={stats['max_ns']/1000:.2f}µs "
                    f"p99={stats['p99_ns']/1000:.2f}µs"
                )
        
        summary.append("\n=== System Metrics ===")
        summary.append(f"CPU: {system['cpu_percent']:.1f}%")
        summary.append(f"Memory: {system['memory_mb']:.1f} MB")
        summary.append(f"Network: ↑{system['network_sent_bytes']/1024:.1f} KB/s "
                      f"↓{system['network_recv_bytes']/1024:.1f} KB/s")
        
        return "\n".join(summary)

# Global instance
latency_monitor = LatencyMonitor()

def measure_latency(metric_name: str):
    """Decorator to measure function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter_ns()
            try:
                return func(*args, **kwargs)
            finally:
                latency_ns = time.perf_counter_ns() - start_time
                latency_monitor.record(metric_name, latency_ns)
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    import random
    
    # Example function with latency measurement
    @measure_latency("example_function")
    def example_function():
        time.sleep(random.uniform(0.001, 0.01))  # 1-10ms delay
    
    # Generate some test data
    print("Generating test data...")
    for _ in range(1000):
        example_function()
    
    # Print summary
    print("\n" + "="*50)
    print(latency_monitor.get_summary())
    print("="*50)
    
    # Example of accessing raw metrics
    metrics = latency_monitor.get_metrics()
    print("\nRaw metrics:")
    print(json.dumps(metrics, indent=2))
