import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy
from ..models import Order, OrderSide, OrderType

class VolatilityBreakout(BaseStrategy):
    """
    Volatility Breakout strategy.
    Buys when price breaks above the previous day's high plus a volatility factor,
    and sells when price breaks below the previous day's low minus a volatility factor.
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        default_config = {
            'atr_period': 14,
            'volatility_factor': 1.0,
            'position_size': 1.0,
            'min_volume': 1000,
            'max_daily_trades': 5
        }
        super().__init__(config or {})
        self.config = {**default_config, **self.config}
        self.data: pd.DataFrame = pd.DataFrame()
        self.daily_data: pd.DataFrame = pd.DataFrame()
        self.today_trades: int = 0
        self.last_trade_day: Optional[pd.Timestamp] = None
        
    async def initialize(self) -> None:
        """Initialize the strategy."""
        self.initialized = True
        
    async def on_market_data(self, data: Dict[str, Any]) -> List[Order]:
        """Process market data and generate trading signals."""
        if not self._validate_market_data(data):
            return []
            
        try:
            self._update_data(data)
            
            if len(self.daily_data) < self.config['atr_period'] + 1:
                return []
                
            return self._generate_signals(data)
        except Exception as e:
            self.logger.error(f"Error in volatility breakout strategy: {str(e)}", exc_info=True)
            return []
    
    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming market data."""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        if not all(field in data for field in required_fields):
            return False
        return True
    
    def _update_data(self, data: Dict[str, Any]) -> None:
        """Update internal data stores with new market data."""
        timestamp = data.get('timestamp', pd.Timestamp.now())
        current_date = pd.Timestamp(timestamp).normalize()
        
        # Reset daily trade count if it's a new day
        if self.last_trade_day != current_date:
            self.today_trades = 0
            self.last_trade_day = current_date
            
        # Update intraday data
        new_row = {
            'timestamp': timestamp,
            'open': float(data['open']),
            'high': float(data['high']),
            'low': float(data['low']),
            'close': float(data['close']),
            'volume': float(data['volume'])
        }
        
        self.data = pd.concat([
            self.data,
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # Update daily data
        self._update_daily_data(new_row, current_date)
    
    def _update_daily_data(self, new_row: Dict[str, Any], current_date: pd.Timestamp) -> None:
        """Update daily OHLCV data."""
        if self.daily_data.empty:
            self.daily_data = pd.DataFrame([{
                'date': current_date,
                'open': new_row['open'],
                'high': new_row['high'],
                'low': new_row['low'],
                'close': new_row['close'],
                'volume': new_row['volume']
            }])
        else:
            last_date = self.daily_data.iloc[-1]['date']
            
            if current_date > last_date:
                # New day, add new row
                self.daily_data = pd.concat([
                    self.daily_data,
                    pd.DataFrame([{
                        'date': current_date,
                        'open': new_row['open'],
                        'high': new_row['high'],
                        'low': new_row['low'],
                        'close': new_row['close'],
                        'volume': new_row['volume']
                    }])
                ], ignore_index=True)
            else:
                # Update current day's data
                idx = self.daily_data.index[-1]
                self.daily_data.at[idx, 'high'] = max(
                    self.daily_data.at[idx, 'high'], 
                    new_row['high']
                )
                self.daily_data.at[idx, 'low'] = min(
                    self.daily_data.at[idx, 'low'], 
                    new_row['low']
                )
                self.daily_data.at[idx, 'close'] = new_row['close']
                self.daily_data.at[idx, 'volume'] += new_row['volume']
    
    def _calculate_atr(self) -> float:
        """Calculate Average True Range."""
        high = self.daily_data['high']
        low = self.daily_data['low']
        close = self.daily_data['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=self.config['atr_period']).mean().iloc[-1]
    
    def _generate_signals(self, data: Dict[str, Any]) -> List[Order]:
        """Generate trading signals based on volatility breakout."""
        if self.today_trades >= self.config['max_daily_trades']:
            return []
            
        current_volume = float(data['volume'])
        if current_volume < self.config['min_volume']:
            return []
            
        atr = self._calculate_atr()
        prev_day = self.daily_data.iloc[-2] if len(self.daily_data) > 1 else None
        current_price = float(data['close'])
        
        if prev_day is None:
            return []
            
        long_breakout = prev_day['high'] + (atr * self.config['volatility_factor'])
        short_breakout = prev_day['low'] - (atr * self.config['volatility_factor'])
        
        orders = []
        
        if current_price > long_breakout:
            self.today_trades += 1
            orders.append(Order(
                symbol=data.get('symbol', 'UNKNOWN'),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.config['position_size']
            ))
        elif current_price < short_breakout:
            self.today_trades += 1
            orders.append(Order(
                symbol=data.get('symbol', 'UNKNOWN'),
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.config['position_size']
            ))
        
        return orders
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for this strategy."""
        base_schema = super().get_config_schema()
        return {
            **base_schema,
            "atr_period": {
                "type": "integer",
                "default": 14,
                "min": 2,
                "max": 50,
                "description": "ATR period"
            },
            "volatility_factor": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "description": "Volatility factor for breakout levels"
            },
            "position_size": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 100.0,
                "description": "Position size in units"
            },
            "min_volume": {
                "type": "float",
                "default": 1000.0,
                "min": 0.0,
                "description": "Minimum volume threshold for trading"
            },
            "max_daily_trades": {
                "type": "integer",
                "default": 5,
                "min": 1,
                "description": "Maximum number of trades per day"
            }
        }