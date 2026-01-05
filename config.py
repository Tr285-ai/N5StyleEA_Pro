# config.py
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")

def get_config() -> Dict[str, Any]:
    """Get configuration with defaults."""
    default_config = {
        "trading": {
            "initial_balance": 10000.0,
            "risk_per_trade": 0.01,
            "max_open_trades": 5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        },
        "data": {
            "data_dir": "data",
            "timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT"],
        },
        "backtest": {
            "initial_balance": 10000.0,
            "commission": 0.001,
            "slippage": 0.0005,
        },
        "logging": {
            "level": "INFO",
            "file": "trading_bot.log",
        },
    }
    
    try:
        config_path = Path("config/config.yaml")
        if config_path.exists():
            user_config = load_config(str(config_path))
            # Deep merge user config with defaults
            import copy
            config = copy.deepcopy(default_config)
            for key, value in user_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
            return config
        return default_config
    except Exception as e:
        print(f"Warning: Using default config due to error: {e}")
        return default_config

def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> None:
    """Save configuration to YAML file."""
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save config to {config_path}: {e}")