# strategies/strategy_factory.py
import importlib
import logging
from typing import Dict, Any, Type, Optional
from .base_strategy import BaseStrategy

logger = logging.getLogger('strategy_factory')

class StrategyFactory:
    """Factory for creating strategy instances."""
    
    _strategy_classes = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]):
        """Register a strategy class."""
        cls._strategy_classes[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    @classmethod
    def create_strategy(cls, name: str, config: Dict[str, Any]) -> Optional[BaseStrategy]:
        """Create a strategy instance by name."""
        strategy_class = cls._strategy_classes.get(name)
        if not strategy_class:
            logger.error(f"Strategy not found: {name}")
            return None
            
        try:
            return strategy_class(config)
        except Exception as e:
            logger.error(f"Failed to create strategy {name}: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, Type[BaseStrategy]]:
        """Get all registered strategy classes."""
        return cls._strategy_classes.copy()
    
    @classmethod
    def load_strategies(cls):
        """Load all strategy modules."""
        strategy_modules = [
            'strategies.moving_average_crossover',
            'strategies.momentum_strategy',
            'strategies.volatility_breakout',
            'strategies.pairs_trading',
            'strategies.hybrid'
        ]
        
        for module_name in strategy_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                logger.warning(f"Failed to load strategy module {module_name}: {str(e)}")

# Auto-load strategies when module is imported
StrategyFactory.load_strategies()