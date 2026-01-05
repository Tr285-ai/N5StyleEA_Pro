import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from .base_strategy import BaseStrategy
from ..models import Order, OrderSide, OrderType
from sklearn.linear_model import LinearRegression

class PairsTrading(BaseStrategy):
    """
    Pairs Trading strategy that identifies correlated pairs of assets
    and trades the spread when it deviates from its historical mean.
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        default_config = {
            'lookback': 20,
            'entry_z': 2.0,
            'exit_z': 0.5,
            'position_size': 1.0,
            'pairs': [],
            'max_position_hold_time': 24  # hours
        }
        super().__init__(config or {})
        self.config = {**default_config, **self.config}
        self.price_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
        self.spreads: Dict[tuple, pd.Series] = {}
        self.positions: Dict[tuple, Dict[str, Any]] = {}
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the strategy."""
        for pair in self.config['pairs']:
            if len(pair) == 2:
                pair_key = self._get_pair_key(pair)
                self.spreads[pair_key] = pd.Series(dtype=float)
                self.positions[pair_key] = {
                    'entry_time': None,
                    'entry_z': 0.0,
                    'position': 0.0
                }
        self.initialized = True
        
    async def on_market_data(self, data: Dict[str, Any]) -> List[Order]:
        """Process market data and generate trading signals."""
        if not self._validate_market_data(data):
            return []
            
        try:
            symbol = data['symbol']
            current_price = float(data['close'])
            timestamp = data.get('timestamp', pd.Timestamp.now())
            
            # Update price history
            self._update_price_history(symbol, timestamp, current_price)
            
            # Check all pairs involving this symbol
            orders = []
            for pair in self.config['pairs']:
                if symbol in pair:
                    pair_orders = await self._process_pair(pair, symbol, current_price, timestamp)
                    orders.extend(pair_orders)
                    
            return orders
        except Exception as e:
            self.logger.error(f"Error in pairs trading strategy: {str(e)}", exc_info=True)
            return []
    
    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming market data."""
        required_fields = ['symbol', 'close']
        return all(field in data for field in required_fields)
    
    def _update_price_history(self, symbol: str, timestamp: pd.Timestamp, price: float) -> None:
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append((timestamp, price))
        
        # Keep only the last N periods to save memory
        max_history = self.config['lookback'] * 2
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
    
    async def _process_pair(self, pair: tuple, symbol: str, current_price: float, 
                          timestamp: pd.Timestamp) -> List[Order]:
        """Process a single pair and generate signals."""
        other_symbol = pair[0] if pair[1] == symbol else pair[1]
        pair_key = self._get_pair_key(pair)
        
        # Check if we have enough data for both symbols
        if (len(self.price_history.get(symbol, [])) < self.config['lookback'] or 
            len(self.price_history.get(other_symbol, [])) < self.config['lookback']):
            return []
            
        # Get price series for both symbols
        prices1 = pd.Series([p[1] for p in self.price_history[symbol][-self.config['lookback']:]])
        prices2 = pd.Series([p[1] for p in self.price_history[other_symbol][-self.config['lookback']:]])
        
        # Calculate spread and z-score
        spread = prices1 - prices2
        zscore = (spread.iloc[-1] - spread.mean()) / (spread.std() + 1e-9)  # Add small value to avoid division by zero
        
        # Update spread history
        self.spreads[pair_key] = pd.concat([
            self.spreads[pair_key],
            pd.Series([zscore], index=[timestamp])
        ])
        
        # Check if we need to exit any positions
        orders = await self._check_exit_conditions(pair_key, symbol, other_symbol, zscore, timestamp)
        if orders:
            return orders
            
        # Check for new entry signals
        return await self._check_entry_conditions(pair_key, symbol, other_symbol, zscore, timestamp)
    
    async def _check_exit_conditions(self, pair_key: tuple, symbol: str, other_symbol: str,
                                   zscore: float, timestamp: pd.Timestamp) -> List[Order]:
        """Check if we need to exit any positions."""
        position = self.positions[pair_key]
        if position['position'] == 0:
            return []
            
        # Check if we've held the position too long
        if (position['entry_time'] is not None and 
            (timestamp - position['entry_time']).total_seconds() > self.config['max_position_hold_time'] * 3600):
            return self._close_position(pair_key, symbol, other_symbol, "timeout")
            
        # Check if spread has reverted to mean
        if ((position['position'] > 0 and zscore <= self.config['exit_z']) or
            (position['position'] < 0 and zscore >= -self.config['exit_z'])):
            return self._close_position(pair_key, symbol, other_symbol, "mean_reversion")
            
        return []
    
    async def _check_entry_conditions(self, pair_key: tuple, symbol: str, other_symbol: str,
                                    zscore: float, timestamp: pd.Timestamp) -> List[Order]:
        """Check for new entry signals."""
        position = self.positions[pair_key]
        if position['position'] != 0:
            return []
            
        orders = []
        
        # Long the spread (buy symbol, sell other_symbol)
        if zscore < -self.config['entry_z']:
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.config['position_size'],
                metadata={'pair_trade': True, 'other_symbol': other_symbol}
            ))
            self._update_position(pair_key, 1, zscore, timestamp)
            
        # Short the spread (sell symbol, buy other_symbol)
        elif zscore > self.config['entry_z']:
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.config['position_size'],
                metadata={'pair_trade': True, 'other_symbol': other_symbol}
            ))
            self._update_position(pair_key, -1, zscore, timestamp)
            
        return orders
    
    def _update_position(self, pair_key: tuple, position: int, zscore: float, 
                        timestamp: pd.Timestamp) -> None:
        """Update position information."""
        self.positions[pair_key] = {
            'entry_time': timestamp,
            'entry_z': zscore,
            'position': position
        }
    
    def _close_position(self, pair_key: tuple, symbol: str, other_symbol: str,
                       reason: str) -> List[Order]:
        """Generate orders to close a position."""
        position = self.positions[pair_key]
        if position['position'] == 0:
            return []
            
        side = OrderSide.SELL if position['position'] > 0 else OrderSide.BUY
        self.positions[pair_key] = {'entry_time': None, 'entry_z': 0.0, 'position': 0}
        
        return [Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=self.config['position_size'],
            metadata={'pair_trade': True, 'action': 'close', 'reason': reason}
        )]
    
    def _get_pair_key(self, pair: tuple) -> tuple:
        """Get a consistent key for a pair regardless of order."""
        return tuple(sorted(pair))
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for this strategy."""
        base_schema = super().get_config_schema()
        return {
            **base_schema,
            "lookback": {
                "type": "integer",
                "default": 20,
                "min": 5,
                "max": 200,
                "description": "Lookback period for z-score calculation"
            },
            "entry_z": {
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "description": "Z-score threshold for entering trades"
            },
            "exit_z": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "description": "Z-score threshold for exiting trades"
            },
            "position_size": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 100.0,
                "description": "Position size in units"
            },
            "pairs": {
                "type": "list",
                "item_type": "tuple",
                "default": [],
                "description": "List of symbol pairs to trade (e.g., [('AAPL', 'MSFT')])"
            },
            "max_position_hold_time": {
                "type": "integer",
                "default": 24,
                "min": 1,
                "description": "Maximum time to hold a position (hours)"
            }
        }