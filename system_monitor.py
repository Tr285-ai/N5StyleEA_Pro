import os
import json
import time
import psutil
import logging
import platform
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class SystemMonitor:
    """
    System monitoring and resource management for N5StyleEA Pro.
    """
    
    def __init__(self, config_path: str = "monitor_config.json"):
        """
        Initialize the system monitor.
        
        Args:
            config_path: Path to the monitor configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("SystemMonitor")
        self.metrics = {}
        self.last_metrics_time = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load monitor configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Monitor config file not found at {config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing monitor config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitor configuration."""
        return {
            "monitoring": {
                "enabled": True,
                "update_interval": 60,
                "max_cpu_percent": 80.0,
                "max_memory_percent": 80.0,
                "max_disk_percent": 90.0,
                "max_temperature": 80.0
            },
            "alerts": {
                "enabled": True,
                "email": {
                    "enabled": False,
                    "recipients": []
                },
                "telegram": {
                    "enabled": False,
                    "bot_token": "",
                    "chat_id": ""
                }
            }
        }
        
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"monitor_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect system metrics.
        
        Returns:
            Dictionary containing system metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq().current if hasattr(psutil, 'cpu_freq') else None
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            # System info
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq,
                    "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                "memory": {
                    "total_gb": round(memory.total / (1024 ** 3), 2),
                    "available_gb": round(memory.available / (1024 ** 3), 2),
                    "used_gb": round(memory.used / (1024 ** 3), 2),
                    "percent": memory.percent,
                    "swap_total_gb": round(swap.total / (1024 ** 3), 2),
                    "swap_used_gb": round(swap.used / (1024 ** 3), 2),
                    "swap_percent": swap.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024 ** 3), 2),
                    "used_gb": round(disk.used / (1024 ** 3), 2),
                    "free_gb": round(disk.free / (1024 ** 3), 2),
                    "percent": disk.percent
                },
                "network": {
                    "bytes_sent_mb": round(net_io.bytes_sent / (1024 ** 2), 2),
                    "bytes_recv_mb": round(net_io.bytes_recv / (1024 ** 2), 2),
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                },
                "system": {
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "hostname": platform.node(),
                    "python_version": platform.python_version(),
                    "boot_time": boot_time.isoformat(),
                    "uptime_seconds": int(uptime.total_seconds())
                }
            }
            
            # Try to get temperature (Linux only)
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        metrics["temperature"] = {
                            "celsius": temps.get('coretemp', [{}])[0].current if 'coretemp' in temps else None
                        }
            except Exception as e:
                self.logger.warning(f"Could not read temperature: {e}")
            
            self.metrics = metrics
            self.last_metrics_time = datetime.now()
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}", exc_info=True)
            return {}
            
    def check_thresholds(self) -> Dict[str, bool]:
        """
        Check if any metrics exceed configured thresholds.
        
        Returns:
            Dictionary of threshold checks with boolean results
        """
        if not self.metrics:
            return {}
            
        checks = {
            "cpu_high": False,
            "memory_high": False,
            "disk_high": False,
            "temperature_high": False
        }
        
        try:
            # CPU check
            if self.metrics["cpu"]["percent"] > self.config["monitoring"]["max_cpu_percent"]:
                checks["cpu_high"] = True
                self.logger.warning(
                    f"High CPU usage: {self.metrics['cpu']['percent']}% "
                    f"(threshold: {self.config['monitoring']['max_cpu_percent']}%)"
                )
                
            # Memory check
            if self.metrics["memory"]["percent"] > self.config["monitoring"]["max_memory_percent"]:
                checks["memory_high"] = True
                self.logger.warning(
                    f"High memory usage: {self.metrics['memory']['percent']}% "
                    f"(threshold: {self.config['monitoring']['max_memory_percent']}%)"
                )
                
            # Disk check
            if self.metrics["disk"]["percent"] > self.config["monitoring"]["max_disk_percent"]:
                checks["disk_high"] = True
                self.logger.warning(
                    f"High disk usage: {self.metrics['disk']['percent']}% "
                    f"(threshold: {self.config['monitoring']['max_disk_percent']}%)"
                )
                
            # Temperature check
            if "temperature" in self.metrics and self.metrics["temperature"].get("celsius"):
                if self.metrics["temperature"]["celsius"] > self.config["monitoring"]["max_temperature"]:
                    checks["temperature_high"] = True
                    self.logger.warning(
                        f"High temperature: {self.metrics['temperature']['celsius']}°C "
                        f"(threshold: {self.config['monitoring']['max_temperature']}°C)"
                    )
                    
        except KeyError as e:
            self.logger.error(f"Error checking thresholds: missing key {e}")
            
        return checks
        
    def send_alert(self, message: str) -> bool:
        """
        Send an alert through configured channels.
        
        Args:
            message: Alert message to send
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        success = True
        
        # Email alert
        if self.config["alerts"].get("email", {}).get("enabled", False):
            try:
                self._send_email_alert(message)
            except Exception as e:
                self.logger.error(f"Failed to send email alert: {e}")
                success = False
                
        # Telegram alert
        if self.config["alerts"].get("telegram", {}).get("enabled", False):
            try:
                self._send_telegram_alert(message)
            except Exception as e:
                self.logger.error(f"Failed to send Telegram alert: {e}")
                success = False
                
        return success
        
    def _send_email_alert(self, message: str) -> None:
        """Send an email alert (implementation depends on email service)."""
        # This is a placeholder - implement based on your email service
        pass
        
    def _send_telegram_alert(self, message: str) -> None:
        """Send a Telegram alert."""
        bot_token = self.config["alerts"]["telegram"].get("bot_token")
        chat_id = self.config["alerts"]["telegram"].get("chat_id")
        
        if not bot_token or not chat_id:
            self.logger.warning("Telegram bot token or chat ID not configured")
            return
            
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": f"🚨 N5StyleEA Alert:\n\n{message}",
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
        except requests.RequestException as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            raise
            
    def run(self) -> None:
        """Run the system monitor in a loop."""
        if not self.config["monitoring"]["enabled"]:
            self.logger.info("System monitoring is disabled")
            return
            
        self.logger.info("Starting system monitor...")
        
        try:
            while True:
                # Collect and log metrics
                self.collect_metrics()
                self.logger.debug(f"System metrics: {json.dumps(self.metrics, indent=2)}")
                
                # Check thresholds and send alerts if needed
                checks = self.check_thresholds()
                if any(checks.values()):
                    alert_message = "Warning: "
                    if checks["cpu_high"]:
                        alert_message += "High CPU usage. "
                    if checks["memory_high"]:
                        alert_message += "High memory usage. "
                    if checks["disk_high"]:
                        alert_message += "High disk usage. "
                    if checks["temperature_high"]:
                        alert_message += "High temperature. "
                        
                    self.send_alert(alert_message.strip())
                    
                # Sleep until next update
                time.sleep(self.config["monitoring"]["update_interval"])
                
        except KeyboardInterrupt:
            self.logger.info("System monitor stopped by user")
        except Exception as e:
            self.logger.critical(f"Fatal error in system monitor: {e}", exc_info=True)
        finally:
            self.logger.info("System monitor stopped")

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.run()