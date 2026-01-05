"""
High-Frequency Trading (HFT) Configuration

This module contains configuration settings for the HFT system.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

@dataclass
class NetworkConfig:
    """Network configuration for HFT system."""
    # Market data feed
    market_data_host: str = "127.0.0.1"
    market_data_port: int = 5000
    multicast_groups: List[str] = None  # For multicast subscriptions
    
    # Order execution
    order_gateway_host: str = "127.0.0.1"
    order_gateway_port: int = 5001
    
    # Network buffer sizes (in bytes)
    recv_buffer_size: int = 1024 * 1024 * 100  # 100MB
    send_buffer_size: int = 1024 * 1024 * 50   # 50MB
    
    # TCP_NODELAY (disable Nagle's algorithm)
    tcp_no_delay: bool = True
    
    # Socket timeout (seconds)
    socket_timeout: float = 0.1

@dataclass
class OrderManagerConfig:
    """Order manager configuration."""
    # Order processing
    enable_jit: bool = True
    batch_size: int = 100
    max_order_queue_size: int = 10000
    
    # Order validation
    max_order_value: float = 1_000_000.0  # Maximum order value in quote currency
    max_position_size: Dict[str, float] = None  # Per-symbol position limits
    
    # Risk limits
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_position_risk: float = 0.1    # 10% of portfolio per position
    max_volatility: float = 0.5       # 50% annualized volatility limit

@dataclass
class MarketDataConfig:
    """Market data feed configuration."""
    # Feed settings
    enable_jit: bool = True
    batch_size: int = 100
    buffer_size: int = 1024 * 1024  # 1MB buffer
    
    # Book depth to maintain
    book_depth: int = 10
    
    # Snapshot and recovery
    snapshot_interval: int = 60  # Seconds between snapshots
    
    # Instrument configuration
    symbols: List[str] = None  # List of symbols to subscribe to

@dataclass
class LatencyConfig:
    """Latency monitoring configuration."""
    enabled: bool = True
    window_size: int = 10000  # Number of samples to keep in memory
    
    # Alert thresholds (nanoseconds)
    alert_thresholds: Dict[str, int] = None
    
    # Percentiles to calculate
    percentiles: List[float] = None

@dataclass
class HFTConfig:
    """Main HFT configuration container."""
    # Component configurations
    network: NetworkConfig = NetworkConfig()
    order_manager: OrderManagerConfig = OrderManagerConfig()
    market_data: MarketDataConfig = MarketDataConfig()
    latency: LatencyConfig = LatencyConfig()
    
    # System settings
    log_level: str = "INFO"
    log_file: str = "hft_system.log"
    
    # Performance tuning
    num_worker_threads: int = 4
    pin_threads_to_cores: bool = True
    
    def __post_init__(self):
        # Initialize default values for nested dataclasses
        if self.market_data.symbols is None:
            self.market_data.symbols = ["BTC-USD", "ETH-USD"]
            
        if self.order_manager.max_position_size is None:
            self.order_manager.max_position_size = {
                "BTC-USD": 100.0,
                "ETH-USD": 1000.0
            }
            
        if self.latency.alert_thresholds is None:
            self.latency.alert_thresholds = {
                "order_processing": 100_000,  # 100µs
                "market_data_processing": 50_000,  # 50µs
                "network_latency": 10_000  # 10µs
            }
            
        if self.latency.percentiles is None:
            self.latency.percentiles = [50, 90, 95, 99, 99.9]

# Default configuration
default_config = HFTConfig()

def load_config(config_file: Optional[str] = None) -> HFTConfig:
    """
    Load configuration from a JSON file or use defaults.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        HFTConfig: Loaded configuration
    """
    if config_file and os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            
        # Create config from JSON
        config = HFTConfig(**config_data)
        
        # Handle nested dataclasses
        for field in ["network", "order_manager", "market_data", "latency"]:
            if field in config_data:
                field_class = globals()[f"{field.capitalize()}Config"]
                setattr(config, field, field_class(**config_data[field]))
                
        return config
    
    return default_config

def save_config(config: HFTConfig, config_file: str):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration to save
        config_file: Path to save configuration file
    """
    import json
    from dataclasses import asdict
    
    # Convert dataclass to dict
    config_dict = asdict(config)
    
    # Save to file
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Create a default config
    config = HFTConfig()
    
    # Save to file
    save_config(config, "hft_config.json")
    
    # Load from file
    loaded_config = load_config("hft_config.json")
    
    print("Configuration saved and loaded successfully!")
    print(f"Market data symbols: {loaded_config.market_data.symbols}")
