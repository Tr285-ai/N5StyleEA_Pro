# config/core_config.py
CORE_CONFIG = {
    "updater": {
        "enabled": True,
        "auto_update": True,
        "check_interval": 3600,  # Check for updates every hour
        "backup_before_update": True,
        "max_backups": 5
    },
    "monitor": {
        "enabled": True,
        "check_interval": 60,  # Check system health every minute
        "max_cpu_percent": 80,
        "max_memory_percent": 80,
        "min_disk_space": 5,  # GB
        "process_name": "python"  # Process to monitor
    },
    "approval": {
        "enabled": True,
        "server_url": "http://localhost:5000/api/approve",
        "api_key": "your_api_key_here",
        "timeout": 30  # seconds
    }
}