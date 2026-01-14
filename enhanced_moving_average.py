import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from .base_strategy import BaseStrategy

class EnhancedMovingAverage(BaseStrategy):
    """
    Enhanced Moving Average Crossover Strategy with additional indicators and risk management.
    Features:
    - Multiple time frame analysis
    - RSI for confirmation
    - MACD for trend confirmation
    - ATR for dynamic stop-loss
    - Position sizing based on volatility
    - Trailing stop-loss
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced strategy.
        
        Config options:
            fast_ma: Period for fast moving average (default: 10)
            slow_ma: Period for slow moving average (default: 30)
            rsi_period: RSI period (default: 14)
            rsi_overbought: RSI overbought threshold (default: 70)
            rsi_oversold: RSI oversold threshold (default: 30)
            atr_period: ATR period for stop-loss (default: 14)
            atr_multiplier: ATR multiplier for stop-loss (default: 2.0)
            risk_per_trade: Risk per trade as % of capital (default: 1.0)
            trailing_stop: Enable trailing stop (default: True)
            trailing_stop_atr_mult: Trailing stop ATR multiplier (default: 1.5)
        """
        default_config = {
            'fast_ma': 10,
            'slow_ma': 30,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 1.0,
            'trailing_stop': True,
            'trailing_stop_atr_mult': 1.5,
            'max_position_size': 0.1,  # Max 10% of capital per trade
            'max_daily_trades': 5,     # Max trades per day
            'max_drawdown_pct': 5.0,   # Max daily drawdown %
            'take_profit_pct': 2.0,    # Take profit as % of entry price
            'stop_loss_pct': 1.0       # Stop loss as % of entry price
        }
        
        # Merge default config with user config
        config = {**default_config, **(config or {})}
        super().__init__(config)
        
        # Initialize state
        self.initialized = True
        self.today_trades = 0
        self.today_pnl = 0.0
        self.last_trade_day = None
        self.current_position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        
        self.logger = logging.getLogger(f'trading.strategy.{self.__class__.__name__}')
        self.logger.info(f"Initialized with config: {config}")
    
    # [Previous implementation methods go here...]
    # (Include all the methods from the previous EnhancedMovingAverageCrossover class)