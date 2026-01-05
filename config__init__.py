import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        return {}

@dataclass
class BacktestConfig:
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    commission: float = 0.001
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BacktestConfig':
        return cls(**data)
        
    def to_dict(self) -> Dict:
        return asdict(self)
        
    def update(self, updates: Dict) -> 'BacktestConfig':
        data = self.to_dict()
        data.update(updates)
        return self.__class__(**data)

@dataclass
class AutotrainConfig:
    model_type: str
    training_data_path: str
    test_size: float = 0.2
    random_state: int = 42
    epochs: int = 100
    batch_size: int = 32
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AutotrainConfig':
        return cls(**data)
        
    def to_dict(self) -> Dict:
        return asdict(self)
        
    def update(self, updates: Dict) -> 'AutotrainConfig':
        data = self.to_dict()
        data.update(updates)
        return self.__class__(**data)