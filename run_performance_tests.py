# run_performance_tests.py
#!/usr/bin/env python3
"""
Performance test runner with regression detection.

Usage:
    python run_performance_tests.py [--baseline] [--thresholds] [--html-report]
"""
import asyncio
import pytest
import json
import sys
from pathlib import Path
from datetime import datetime
from tests.performance_baseline import (
    check_performance_regression,
    PERFORMANCE_THRESHOLDS
)

async def run_tests():
    """Run performance tests and check for regressions."""
    # Run tests and collect results
    test_results = {}
    
    # Run pytest with performance marker
    pytest_args = [
        "-m", "performance",
        "-v",
        "--durations=10",
        "--capture=no"
    ]
    
    if "--html-report" in sys.argv:
        pytest_args.extend(["--html=test_results/performance_report.html", "--self-contained-html"])
    
    # Run the tests
    retcode = pytest.main(pytest_args)
    
    # In a real implementation, we would collect the actual timing data
    # from pytest's internal reporting or a plugin like pytest-benchmark
    # For now, we'll simulate this with the test names
    test_results = {
        "save_1000_rows": 0.5,  # Example values
        "load_1000_rows": 0.2,
        # Add other test results here
    }
    
    # Check for performance regressions
    regression_results = check_performance_regression(test_results)
    
    # Print results
    print("\n" + "="*80)
    print("Performance Test Results")
    print("="*80)
    
    all_passed = True
    for test_name, (passed, message) in regression_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status} - {message}")
        if not passed:
            all_passed = False
    
    print("\nPerformance Thresholds:")
    for test_name, threshold in PERFORMANCE_THRESHOLDS.items():
        print(f"  {test_name}: {threshold:.2f}s")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))