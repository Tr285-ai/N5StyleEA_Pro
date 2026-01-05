# utils/memory_profiler.py
import tracemalloc
import time
from functools import wraps
from typing import Callable, Dict, List, Tuple
import pandas as pd

class MemoryProfiler:
    """Memory profiler for tracking memory usage."""
    
    def __init__(self):
        self.snapshots = {}
        self.start_time = None
        
    def start(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.start_time = time.time()
        self.snapshots = {
            'timestamps': [],
            'memory_usage': [],
            'snapshots': []
        }
        
    def take_snapshot(self, label: str = None):
        """Take a memory snapshot."""
        if not self.start_time:
            self.start()
            
        snapshot = tracemalloc.take_snapshot()
        current_mem = tracemalloc.get_traced_memory()[0] / 1024**2  # MB
        
        self.snapshots['timestamps'].append(time.time() - self.start_time)
        self.snapshots['memory_usage'].append(current_mem)
        self.snapshots['snapshots'].append((label, snapshot))
        
        return current_mem
        
    def stop(self):
        """Stop memory profiling and generate report."""
        tracemalloc.stop()
        return self.generate_report()
        
    def generate_report(self) -> Dict:
        """Generate memory usage report."""
        if not self.snapshots['timestamps']:
            return {}
            
        report = {
            'peak_memory_mb': max(self.snapshots['memory_usage']),
            'average_memory_mb': sum(self.snapshots['memory_usage']) / len(self.snapshots['memory_usage']),
            'duration_seconds': self.snapshots['timestamps'][-1] if self.snapshots['timestamps'] else 0,
            'memory_leaks': self._find_memory_leaks()
        }
        
        return report
        
    def _find_memory_leaks(self) -> List[Dict]:
        """Identify potential memory leaks by comparing snapshots."""
        leaks = []
        snapshots = self.snapshots['snapshots']
        
        if len(snapshots) < 2:
            return []
            
        for i in range(1, len(snapshots)):
            prev_label, prev_snapshot = snapshots[i-1]
            curr_label, curr_snapshot = snapshots[i]
            
            top_stats = curr_snapshot.compare_to(prev_snapshot, 'lineno')
            
            for stat in top_stats[:10]:  # Top 10 memory consumers
                if stat.size_diff > 0:  # Only care about increasing memory
                    leaks.append({
                        'between': f"{prev_label} -> {curr_label}",
                        'file': stat.traceback[0].filename,
                        'line': stat.traceback[0].lineno,
                        'size_diff_kb': stat.size_diff / 1024,
                        'total_size_kb': stat.size / 1024
                    })
                    
        return leaks

def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.start()
        
        try:
            result = func(*args, **kwargs)
            report = profiler.stop()
            print(f"Memory usage report for {func.__name__}:")
            print(f"  Peak memory: {report['peak_memory_mb']:.2f} MB")
            print(f"  Average memory: {report['average_memory_mb']:.2f} MB")
            print(f"  Duration: {report['duration_seconds']:.2f} seconds")
            
            if report['memory_leaks']:
                print("\nPotential memory leaks:")
                for leak in report['memory_leaks'][:5]:  # Show top 5
                    print(f"  - {leak['between']}: {leak['file']}:{leak['line']} "
                          f"(+{leak['size_diff_kb']:.2f} KB)")
            
            return result
        except Exception as e:
            profiler.stop()
            raise e
            
    return wrapper