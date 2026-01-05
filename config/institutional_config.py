from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import yaml
import os
from pathlib import Path

@dataclass
class RiskConfig:
    """Configuration for institutional risk management"""
    max_position_size: float = 0.1  # % of portfolio
    max_daily_loss: float = 0.02   # 2% max daily loss
    max_drawdown: float = 0.1      # 10% max drawdown
    var_confidence: float = 0.95   # 95% confidence level
    stress_scenarios: List[Dict] = None
    
    def __post_init__(self):
        if self.stress_scenarios is None:
            self.stress_scenarios = [
                {"name": "flash_crash", "price_drop": 0.2},
                {"name": "volatility_spike", "vol_multiplier": 3.0},
                {"name": "liquidity_crunch", "spread_increase": 0.001}
            ]

@dataclass
class HFTConfig:
    """Configuration for HFT optimizations"""
    enabled: bool = True
    use_udp: bool = True
    max_orders_per_second: int = 100000
    latency_budget_ns: int = 100000  # 100 microseconds
    jit_compile: bool = True
    cache_pre_warming: bool = True
    
@dataclass
class NetworkConfig:
    """Network optimization settings"""
    tcp_nodelay: bool = True
    so_priority: int = 6
    receive_buffer_size: int = 65536
    send_buffer_size: int = 65536
    use_zerocopy: bool = True

@dataclass
class InstitutionalConfig:
    """Main configuration class for institutional features"""
    risk: RiskConfig
    hft: HFTConfig
    network: NetworkConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InstitutionalConfig':
        """Create config from dictionary"""
        return cls(
            risk=RiskConfig(**config_dict.get('risk', {})),
            hft=HFTConfig(**config_dict.get('hft', {})),
            network=NetworkConfig(**config_dict.get('network', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'risk': asdict(self.risk),
            'hft': asdict(self.hft),
            'network': asdict(self.network)
        }
    
    @classmethod
    def load(cls, config_path: str = None) -> 'InstitutionalConfig':
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent, 'institutional_config.yaml')
            
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        else:
            # Return default config if file doesn't exist
            return cls.get_default()
            
        return cls.from_dict(config_dict)
    
    def save(self, config_path: str = None):
        """Save configuration to YAML file"""
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent, 'institutional_config.yaml')
            
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def get_default(cls) -> 'InstitutionalConfig':
        """Get default configuration"""
        return cls(
            risk=RiskConfig(),
            hft=HFTConfig(),
            network=NetworkConfig()
        )

# Example usage
if __name__ == "__main__":
    # Create and save default config
    config = InstitutionalConfig.get_default()
    config.save()
    
    # Load config
    loaded_config = InstitutionalConfig.load()
    print("Loaded config:", loaded_config.to_dict())
