from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Supported timeframes for backtesting."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

class DataSource(Enum):
    """Supported data sources for backtesting."""
    LOCAL = "local"
    YAHOO = "yahoo"
    ALPACA = "alpaca"
    BINANCE = "binance"
    COINBASE = "coinbase"
    CCXT = "ccxt"

@dataclass
class BacktestConfig:
    """Configuration for backtesting a trading strategy."""
    # Basic settings
    strategy_name: str
    symbol: str
    timeframe: Union[str, Timeframe]
    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    
    # Date range
    start_date: Optional[Union[str, datetime]] = None
    end_date: Optional[Union[str, datetime]] = None
    lookback_days: Optional[int] = 365  # Used if start_date is not provided
    
    # Data source
    data_source: Union[str, DataSource] = DataSource.LOCAL
    data_path: Optional[str] = None
    
    # Strategy parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Risk management
    position_sizing: float = 0.95  # % of balance to use per trade
    max_drawdown: Optional[float] = None  # Stop backtest if drawdown exceeds this %
    max_trades: Optional[int] = None  # Maximum number of trades to execute
    
    # Performance metrics
    benchmark: str = "SPY"  # Benchmark for comparison
    risk_free_rate: float = 0.0  # Risk-free rate for Sharpe ratio
    
    # Output settings
    output_dir: str = "backtest_results"
    save_trades: bool = True
    save_equity_curve: bool = True
    save_metrics: bool = True
    plot_results: bool = True
    
    def __post_init__(self):
        """Validate and convert configuration values."""
        # Convert string timeframes to Timeframe enum
        if isinstance(self.timeframe, str):
            try:
                self.timeframe = Timeframe(self.timeframe.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid timeframe: {self.timeframe}. "
                    f"Must be one of: {[t.value for t in Timeframe]}"
                )
        
        # Convert string data source to DataSource enum
        if isinstance(self.data_source, str):
            try:
                self.data_source = DataSource(self.data_source.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid data source: {self.data_source}. "
                    f"Must be one of: {[s.value for s in DataSource]}"
                )
        
        # Convert string dates to datetime
        if isinstance(self.start_date, str):
            self.start_date = datetime.fromisoformat(self.start_date)
        elif self.start_date is None:
            self.start_date = datetime.now() - timedelta(days=self.lookback_days)
            
        if isinstance(self.end_date, str):
            self.end_date = datetime.fromisoformat(self.end_date)
        elif self.end_date is None:
            self.end_date = datetime.now()
        
        # Validate date range
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
            
        # Validate numeric values
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not (0 <= self.commission <= 1):
            raise ValueError("commission must be between 0 and 1")
        if not (0 <= self.slippage <= 1):
            raise ValueError("slippage must be between 0 and 1")
        if not (0 < self.position_sizing <= 1):
            raise ValueError("position_sizing must be between 0 and 1")
        if self.max_drawdown is not None and not (0 <= self.max_drawdown <= 1):
            raise ValueError("max_drawdown must be between 0 and 1")
        if self.max_trades is not None and self.max_trades < 1:
            raise ValueError("max_trades must be at least 1")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """Create a BacktestConfig from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'BacktestConfig':
        """Load configuration from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe.value,
            'initial_balance': self.initial_balance,
            'commission': self.commission,
            'slippage': self.slippage,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'data_source': self.data_source.value,
            'data_path': str(self.data_path) if self.data_path else None,
            'params': self.params,
            'position_sizing': self.position_sizing,
            'max_drawdown': self.max_drawdown,
            'max_trades': self.max_trades,
            'benchmark': self.benchmark,
            'risk_free_rate': self.risk_free_rate,
            'output_dir': self.output_dir,
            'save_trades': self.save_trades,
            'save_equity_curve': self.save_equity_curve,
            'save_metrics': self.save_metrics,
            'plot_results': self.plot_results
        }
    
    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        try:
            config_dict = self.to_dict()
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved configuration to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {e}")
            raise
    
    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            self.__post_init__()
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False