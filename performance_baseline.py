# tests/performance_baseline.py
"""
Performance baseline configuration.

This file defines acceptable performance thresholds for tests.
If a test exceeds these thresholds, it will be marked as failed.
"""
import os
from pathlib import Path

# Base directory for performance data
PERF_DATA_DIR = Path("test_data/performance")
PERF_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Performance thresholds (in seconds)
PERFORMANCE_THRESHOLDS = {
    # Format: "test_name": max_seconds
    "save_1000_rows": 1.0,
    "load_1000_rows": 0.5,
    "save_10000_rows": 5.0,
    "load_10000_rows": 1.0,
    "save_100000_rows": 20.0,
    "load_100000_rows": 5.0,
    "concurrent_1_workers": 2.0,
    "concurrent_2_workers": 3.0,
    "concurrent_4_workers": 4.0,
    "concurrent_8_workers": 6.0,
    "large_dataset_save": 30.0,
    "large_dataset_load": 10.0,
}

def save_benchmark_results(results: dict):
    """Save benchmark results to a JSON file with timestamp."""
    import json
    from datetime import datetime
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = PERF_DATA_DIR / f"benchmark_{timestamp}.json"
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    # Keep only the last 10 benchmark files
    cleanup_old_benchmarks()
    
    return filepath

def load_latest_benchmark():
    """Load the most recent benchmark results."""
    import json
    
    benchmarks = list(PERF_DATA_DIR.glob("benchmark_*.json"))
    if not benchmarks:
        return None
        
    latest = max(benchmarks, key=os.path.getmtime)
    with open(latest, "r") as f:
        return json.load(f)

def cleanup_old_benchmarks(keep=10):
    """Remove old benchmark files, keeping only the most recent ones."""
    benchmarks = sorted(PERF_DATA_DIR.glob("benchmark_*.json"), 
                       key=os.path.getmtime)
    
    for old_file in benchmarks[:-keep]:
        try:
            old_file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete {old_file}: {e}")

def check_performance_regression(current_results: dict) -> dict:
    """
    Check if current performance is within acceptable thresholds.
    
    Returns:
        dict: Dictionary with test names as keys and (passed, message) as values
    """
    baseline = load_latest_benchmark()
    if baseline is None:
        print("No baseline found. Creating initial baseline.")
        save_benchmark_results(current_results)
        return {test: (True, "Initial baseline created") for test in current_results}
    
    results = {}
    
    for test_name, current_time in current_results.items():
        # Check against threshold
        threshold = PERFORMANCE_THRESHOLDS.get(test_name)
        if threshold and current_time > threshold:
            results[test_name] = (
                False, 
                f"Performance regression: {current_time:.2f}s > {threshold:.2f}s threshold"
            )
            continue
            
        # Check against baseline (if available)
        baseline_time = baseline.get(test_name, {}).get("time_seconds")
        if baseline_time:
            if current_time > baseline_time * 1.5:  # 50% slower
                results[test_name] = (
                    False,
                    f"Performance regression: {current_time:.2f}s > 1.5x baseline {baseline_time:.2f}s"
                )
                continue
                
        results[test_name] = (True, "Performance OK")
    
    # Save current results as new baseline if all tests pass
    if all(passed for passed, _ in results.values()):
        save_benchmark_results(current_results)
    
    return results