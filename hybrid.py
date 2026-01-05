import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from .base_strategy import BaseStrategy
from ..models import Order, OrderSide, OrderType

class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy that combines multiple strategies and uses a voting system
    to make trading decisions.
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        default_config = {
            'strategies': [],
            'voting_threshold': 0.5,
            'strategy_weights': {},
            'max_position_size': 0.1,
            'max_daily_trades': 10
        }
        super().__init__(config or {})
        self.config = {**default_config, **self.config}
        self.strategy_instances: Dict[str, BaseStrategy] = {}
        self.today_trades: int = 0
        self.last_trade_day: Optional[str] = None
        self.initialized = False
        self.logger = logging.getLogger('trading.strategy.hybrid')
        
    async def initialize(self) -> None:
        """Initialize the hybrid strategy and all sub-strategies."""
        if self.initialized:
            return
            
        try:
            # Initialize all sub-strategies
            for strategy_config in self.config['strategies']:
                await self._initialize_strategy(strategy_config)
                
            self.initialized = True
            self.logger.info(f"Initialized HybridStrategy with {len(self.strategy_instances)} sub-strategies")
        except Exception as e:
            self.logger.error(f"Failed to initialize HybridStrategy: {str(e)}", exc_info=True)
            raise
    
    async def _initialize_strategy(self, strategy_config: Dict[str, Any]) -> None:
        """Initialize a single sub-strategy."""
        strategy_name = strategy_config.get('name')
        if not strategy_name:
            self.logger.warning("Skipping strategy with no name")
            return
            
        try:
            # Dynamic import of strategy class
            module_name = f"trading_system.strategies.{strategy_name.lower()}"
            module = __import__(module_name, fromlist=[strategy_name])
            strategy_class = getattr(module, strategy_name)
            
            # Create and initialize strategy instance
            strategy_instance = strategy_class(strategy_config.get('params', {}))
            await strategy_instance.initialize()
            
            # Store the instance
            self.strategy_instances[strategy_name] = strategy_instance
            self.logger.info(f"Initialized strategy: {strategy_name}")
            
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import strategy {strategy_name}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error initializing strategy {strategy_name}: {str(e)}", exc_info=True)
    
    async def on_market_data(self, data: Dict[str, Any]) -> List[Order]:
        """Process market data through all strategies and combine signals."""
        if not self.initialized or not self.strategy_instances:
            return []
            
        try:
            # Reset daily metrics if needed
            self._reset_daily_metrics(data.get('timestamp'))
            
            # Check if we've hit daily trade limit
            if self.today_trades >= self.config['max_daily_trades']:
                return []
                
            # Collect signals from all strategies
            all_orders = await self._collect_strategy_signals(data)
            
            # Apply voting system
            return self._apply_voting_system(all_orders)
            
        except Exception as e:
            self.logger.error(f"Error in HybridStrategy on_market_data: {str(e)}", exc_info=True)
            return []
    
    async def _collect_strategy_signals(self, data: Dict[str, Any]) -> List[Tuple[Order, str, float]]:
        """Collect trading signals from all strategies."""
        all_orders = []
        
        for strategy_name, strategy in self.strategy_instances.items():
            try:
                # Get orders from this strategy
                orders = await strategy.on_market_data(data)
                if not orders:
                    continue
                    
                # Get weight for this strategy (default to 1.0 if not specified)
                weight = self.config['strategy_weights'].get(strategy_name, 1.0)
                
                # Add to all orders with strategy info
                for order in orders:
                    all_orders.append((order, strategy_name, weight))
                    
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy_name}: {str(e)}", exc_info=True)
        
        return all_orders
    
    def _apply_voting_system(self, all_orders: List[Tuple[Order, str, float]]) -> List[Order]:
        """Apply voting system to determine final orders."""
        if not all_orders:
            return []
            
        # Group orders by (symbol, side)
        order_groups: Dict[Tuple[str, str], List[Tuple[Order, float]]] = {}
        
        for order, strategy_name, weight in all_orders:
            key = (order.symbol, order.side.value)
            if key not in order_groups:
                order_groups[key] = []
            order_groups[key].append((order, weight))
        
        # Apply voting threshold
        final_orders = []
        total_weight = sum(w for orders in order_groups.values() for _, w in orders)
        threshold = self.config['voting_threshold'] * total_weight if total_weight > 0 else 0
        
        for (symbol, side), order_weights in order_groups.items():
            total_votes = sum(w for _, w in order_weights)
            
            if total_votes >= threshold: