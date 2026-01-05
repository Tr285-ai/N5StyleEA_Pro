from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..models import Order

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Dictionary containing strategy configuration
        """
        self.config = config or {}
        self.initialized = False
        self.logger = None  # Logger should be set by the strategy manager
        
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the strategy. This should be overridden by subclasses.
        """
        pass
        
    @abstractmethod
    async def on_market_data(self, data: Dict[str, Any]) -> List[Order]:
        """
        Process market data and generate trading signals.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of Order objects
        """
        pass
        
    async def on_order_update(self, order: Order) -> None:
        """
        Handle order updates. Can be overridden by subclasses.
        
        Args:
            order: Updated order
        """
        pass
        
    async def on_position_update(self, position: Dict[str, Any]) -> None:
        """
        Handle position updates. Can be overridden by subclasses.
        
        Args:
            position: Updated position
        """
        pass
        
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this strategy.
        
        Returns:
            Dictionary containing the configuration schema
        """
        return {
            "enabled": {
                "type": "boolean",
                "default": True,
                "description": "Enable/disable the strategy"
            }
        }
        
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        schema = self.get_config_schema()
        if not isinstance(schema, dict):
            self.logger.error("Invalid configuration schema: must be a dictionary")
            return False
            
        for key, value in self.config.items():
            if key not in schema:
                self.logger.warning(f"Unknown configuration key: {key}")
                continue
                
            param_schema = schema[key]
            param_type = param_schema.get('type')
            param_value = value
            
            # Type checking
            if param_type == 'integer' and not isinstance(param_value, int):
                self.logger.error(f"Invalid type for {key}: expected integer, got {type(param_value).__name__}")
                return False
            elif param_type == 'float' and not isinstance(param_value, (int, float)):
                self.logger.error(f"Invalid type for {key}: expected float, got {type(param_value).__name__}")
                return False
            elif param_type == 'boolean' and not isinstance(param_value, bool):
                self.logger.error(f"Invalid type for {key}: expected boolean, got {type(param_value).__name__}")
                return False
            elif param_type == 'string' and not isinstance(param_value, str):
                self.logger.error(f"Invalid type for {key}: expected string, got {type(param_value).__name__}")
                return False
                
            # Range checking for numeric types
            if param_type in ['integer', 'float']:
                min_val = param_schema.get('min')
                max_val = param_schema.get('max')
                
                if min_val is not None and param_value < min_val:
                    self.logger.error(f"Value for {key} ({param_value}) is below minimum ({min_val})")
                    return False
                    
                if max_val is not None and param_value > max_val:
                    self.logger.error(f"Value for {key} ({param_value}) is above maximum ({max_val})")
                    return False
                    
        return True