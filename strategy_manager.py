import logging
from typing import Dict, List, Type, Any, Optional
from importlib import import_module

from .base_strategy import BaseStrategy
from .models import Order, Position

logger = logging.getLogger('trading.strategy.manager')

class StrategyManager:
    """Manages multiple trading strategies and their execution."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy manager.
        
        Args:
            config: Configuration dictionary with strategy settings
        """
        self.logger = logging.getLogger('trading.strategy.manager')
        self.strategies: Dict[str, BaseStrategy] = {}
        self.config = config
        self.initialized = False
    
    def initialize_strategies(self) -> bool:
        """Initialize all configured strategies."""
        if self.initialized:
            return True
            
        self.logger.info("Initializing strategies...")
        
        # Load strategy configurations
        strategies_config = self.config.get('strategies', {})
        
        # Initialize each strategy
        for strategy_name, strategy_config in strategies_config.items():
            if not strategy_config.get('enabled', True):
                self.logger.info(f"Skipping disabled strategy: {strategy_name}")
                continue
                
            try:
                # Dynamically import strategy module
                module_path = f"strategies.{strategy_name.lower()}"
                module = import_module(module_path)
                
                # Get strategy class (assuming class name follows StrategyName convention)
                class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy'
                strategy_class = getattr(module, class_name)
                
                # Initialize the strategy
                strategy = strategy_class(strategy_config)
                strategy.initialize()
                self.strategies[strategy_name] = strategy
                self.logger.info(f"Initialized strategy: {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize strategy {strategy_name}: {str(e)}", exc_info=True)
        
        self.initialized = bool(self.strategies)
        return self.initialized
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using all active strategies.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary of signals from all strategies
        """
        if not self.initialized:
            self.logger.warning("Strategy manager not initialized")
            return {}
            
        signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signals[strategy_name] = await strategy.on_market_data(market_data)
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy_name}: {str(e)}", exc_info=True)
        
        return signals
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all registered strategies."""
        return self.strategies.copy()
    
    async def on_order_update(self, order: Order) -> None:
        """Notify all strategies about order updates."""
        for strategy in self.strategies.values():
            try:
                await strategy.on_order_update(order)
            except Exception as e:
                self.logger.error(f"Error in on_order_update: {str(e)}", exc_info=True)
    
    async def on_position_update(self, position: Position) -> None:
        """Notify all strategies about position updates."""
        for strategy in self.strategies.values():
            try:
                await strategy.on_position_update(position)
            except Exception as e:
                self.logger.error(f"Error in on_position_update: {str(e)}", exc_info=True)