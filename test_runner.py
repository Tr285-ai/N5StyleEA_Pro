# test_runner.py
import unittest
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_runner')

class TestRunner:
    """
    Enhanced test runner for the trading system.
    Handles discovery, execution, and reporting of all tests.
    """
    
    def __init__(self, test_dir: str = "tests", output_dir: str = "test_results"):
        """
        Initialize the test runner.
        
        Args:
            test_dir: Directory containing test files
            output_dir: Directory to save test results
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_suites": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "success_rate": 0.0
            }
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def discover_tests(self) -> List[unittest.TestSuite]:
        """Discover all test cases in the test directory."""
        loader = unittest.TestLoader()
        return loader.discover(str(self.test_dir), pattern="test_*.py")
        
    def run_tests(self) -> Dict[str, Any]:
        """Run all discovered tests and return results."""
        test_suites = self.discover_tests()
        runner = unittest.TextTestRunner(verbosity=2)
        
        self.results["summary"]["total"] = test_suites.countTestCases()
        
        for suite in test_suites:
            suite_result = {
                "name": str(suite),
                "start_time": datetime.utcnow().isoformat(),
                "tests": [],
                "summary": {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "error": 0
                }
            }
            
            # Run the test suite
            result = runner.run(suite)
            
            # Process results
            for test_case in result.tests:
                test_name = test_case.id()
                test_status = self._get_test_status(test_case, result)
                test_duration = 0.0  # Can be enhanced with timing
                
                test_result = {
                    "name": test_name,
                    "status": test_status,
                    "duration": test_duration,
                    "message": ""
                }
                
                # Update counters
                suite_result["summary"]["total"] += 1
                suite_result["summary"][test_status] += 1
                self.results["summary"][test_status] += 1
                
                suite_result["tests"].append(test_result)
                
            suite_result["end_time"] = datetime.utcnow().isoformat()
            self.results["test_suites"].append(suite_result)
        
        # Calculate success rate
        total = self.results["summary"]["total"]
        if total > 0:
            passed = self.results["summary"]["passed"]
            self.results["summary"]["success_rate"] = (passed / total) * 100
            
        return self.results
        
    def _get_test_status(self, test_case, result) -> str:
        """Determine the status of a test case."""
        if test_case in result.failures or test_case in result.errors:
            return "failed" if test_case in result.failures else "error"
        return "passed"
        
    def generate_report(self) -> str:
        """Generate a test report in JSON format."""
        report_path = self.output_dir / f"test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        return str(report_path)
        
    def run(self) -> bool:
        """Run all tests and generate a report."""
        logger.info("Starting test execution...")
        self.run_tests()
        report_path = self.generate_report()
        
        logger.info(f"Test execution complete. Report saved to: {report_path}")
        logger.info(f"Summary: {self.results['summary']}")
        
        return self.results['summary']['failed'] == 0 and self.results['summary']['error'] == 0

if __name__ == "__main__":
    # Run all tests
    runner = TestRunner(test_dir=".", output_dir="test_results")
    success = runner.run()
    
    exit(0 if success else 1)