#!/usr/bin/env python3
"""
Performance monitoring script for the trading system.
Monitors CPU, memory, and disk usage at regular intervals.
"""

import argparse
import time
import csv
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party imports
try:
    import psutil
    import pandas as pd
except ImportError as e:
    print(f"Error: Required packages not found. Please install them using:")
    print("pip install psutil pandas")
    sys.exit(1)

# Constants
DEFAULT_DURATION = 300  # 5 minutes
DEFAULT_INTERVAL = 5    # 5 seconds
OUTPUT_DIR = "performance_logs"
OUTPUT_FILE = "performance_metrics.csv"

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor system performance metrics.")
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Monitoring duration in seconds (default: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Sampling interval in seconds (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save output files (default: {OUTPUT_DIR})"
    )
    return parser.parse_args()


def get_system_metrics() -> Dict[str, Any]:
    """Collect system performance metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024 ** 3),
        "memory_available_gb": memory.available / (1024 ** 3),
        "disk_percent": disk.percent,
        "disk_used_gb": disk.used / (1024 ** 3),
        "disk_free_gb": disk.free / (1024 ** 3),
    }


def save_metrics_to_csv(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics to a CSV file."""
    file_exists = os.path.isfile(filepath)
    
    try:
        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    except IOError as e:
        logger.error(f"Error writing to {filepath}: {e}")
        raise


def monitor_performance(duration: int, interval: int, output_dir: str) -> None:
    """Monitor system performance for the specified duration and interval."""
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, OUTPUT_FILE)
        
        logger.info(f"Starting performance monitoring for {duration} seconds (interval: {interval}s)")
        logger.info(f"Output will be saved to: {output_file}")
        logger.info("Press Ctrl+C to stop monitoring early...")
        
        start_time = time.time()
        end_time = start_time + duration
        iteration = 0
        
        try:
            while time.time() < end_time:
                iteration += 1
                metrics = get_system_metrics()
                save_metrics_to_csv(metrics, output_file)
                
                # Log status
                logger.info(
                    f"Iteration {iteration} - "
                    f"CPU: {metrics['cpu_percent']:.1f}% | "
                    f"Memory: {metrics['memory_percent']:.1f}% | "
                    f"Disk: {metrics['disk_percent']:.1f}%"
                )
                
                # Sleep for the interval, but check for early termination
                time_to_sleep = min(interval, end_time - time.time())
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    break
                    
        except KeyboardInterrupt:
            logger.info("\nMonitoring stopped by user.")
        except Exception as e:
            logger.error(f"Error during monitoring: {e}", exc_info=True)
        finally:
            logger.info(f"\nMonitoring complete. Data saved to {output_file}")
            
            # Generate a summary report
            try:
                generate_summary_report(output_file)
            except Exception as e:
                logger.error(f"Could not generate summary report: {e}")
    
    except Exception as e:
        logger.critical(f"Fatal error in monitor_performance: {e}", exc_info=True)
        raise


def generate_summary_report(csv_file: str) -> None:
    """Generate a summary report from the collected metrics."""
    try:
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}")
            return
            
        df = pd.read_csv(csv_file)
        if df.empty:
            logger.warning("No data collected for summary report.")
            return
                
        summary_file = os.path.splitext(csv_file)[0] + "_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== Performance Monitoring Summary ===\n")
            f.write(f"Generated at: {datetime.utcnow().isoformat()}\n")
            f.write(f"Total samples: {len(df)}\n\n")
            
            f.write("CPU Usage (%):\n")
            f.write(f"  Average: {df['cpu_percent'].mean():.1f}%\n")
            f.write(f"  Maximum: {df['cpu_percent'].max():.1f}%\n")
            f.write(f"  Minimum: {df['cpu_percent'].min():.1f}%\n\n")
            
            f.write("Memory Usage (%):\n")
            f.write(f"  Average: {df['memory_percent'].mean():.1f}%\n")
            f.write(f"  Maximum: {df['memory_percent'].max():.1f}%\n")
            f.write(f"  Minimum: {df['memory_percent'].min():.1f}%\n\n")
            
            f.write("Disk Usage (%):\n")
            f.write(f"  Average: {df['disk_percent'].mean():.1f}%\n")
            f.write(f"  Maximum: {df['disk_percent'].max():.1f}%\n")
            f.write(f"  Minimum: {df['disk_percent'].min():.1f}%\n")
            
        logger.info(f"Summary report generated: {summary_file}")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        args = parse_arguments()
        monitor_performance(
            duration=args.duration,
            interval=args.interval,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)