from typing import Dict, Type, Optional, Any
import logging
from .base_broker import BaseBroker
from .pocketoption import PocketOptionBroker

logger = logging.getLogger('broker_manager')

@dataclass
class BrokerConfig:
    """Configuration for a broker connection."""
    name: str
    enabled: bool
    config: Dict[str, Any]

class BrokerManager:
    """Manages multiple broker connections."""
    
    def __init__(self, broker_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize the broker manager.
        
        Args:
            broker_configs: Dictionary of broker configurations
        """
        self.brokers: Dict[str, BaseBroker] = {}
        self.configs: Dict[str, BrokerConfig] = {}
        
        # Parse configurations
        for name, config in broker_configs.items():
            self.configs[name] = BrokerConfig(
                name=name,
                enabled=config.get('enabled', False),
                config=config
            )
    
    async def initialize(self) -> bool:
        """Initialize all enabled brokers."""
        logger.info("Initializing brokers...")
        
        for name, config in self.configs.items():
            if not config.enabled:
                logger.info(f"Skipping disabled broker: {name}")
                continue
                
            try:
                broker = await self._create_broker(name, config.config)
                if broker:
                    self.brokers[name] = broker
                    logger.info(f"Successfully initialized broker: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize broker {name}: {str(e)}", exc_info=True)
        
        return len(self.brokers) > 0
    
    async def _create_broker(self, name: str, config: Dict[str, Any]) -> Optional[BaseBroker]:
        """Create a broker instance based on configuration."""
        broker_type = config.get('type', name).lower()
        
        if broker_type == 'pocketoption':
            return await self._create_pocketoption_broker(config)
        else:
            logger.warning(f"Unknown broker type: {broker_type}")
            return None
    
    async def _create_pocketoption_broker(self, config: Dict[str, Any]) -> Optional[BaseBroker]:
        """Create a PocketOption broker instance."""
        required = ['email', 'password', 'account_id']
        if not all(k in config for k in required):
            logger.error(f"Missing required configuration for PocketOption: {required}")
            return None
            
        broker = PocketOptionBroker(
            email=config['email'],
            password=config['password'],
            account_id=config['account_id'],
            is_testnet=config.get('is_testnet', False)
        )
        
        connected = await broker.connect()
        if not connected:
            logger.error("Failed to connect to PocketOption")
            return None
            
        return broker
    
    async def shutdown(self) -> None:
        """Shut down all broker connections."""
        logger.info("Shutting down brokers...")
        
        for name, broker in self.brokers.items():
            try:
                await broker.disconnect()
                logger.info(f"Successfully disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {str(e)}", exc_info=True)
        
        self.brokers.clear()
    
    def get_broker(self, name: str) -> Optional[BaseBroker]:
        """Get a broker instance by name."""
        return self.brokers.get(name)

class BrokerFactory:
    """Factory class for creating broker instances."""
    
    _brokers: Dict[str, Type[BaseBroker]] = {
        'pocketoption': PocketOptionBroker,
        # Add other brokers here
    }
    
    @classmethod
    def create_broker(cls, broker_type: str, config: Dict[str, Any]) -> BaseBroker:
        """Create a broker instance by type."""
        broker_class = cls._brokers.get(broker_type.lower())
        if not broker_class:
            raise ValueError(f"Unsupported broker type: {broker_type}")
        return broker_class(**config)
    
    @classmethod
    def register_broker(cls, name: str, broker_class: Type[BaseBroker]) -> None:
        """Register a new broker type."""
        if not issubclass(broker_class, BaseBroker):
            raise TypeError("Broker must be a subclass of BaseBroker")
        cls._brokers[name.lower()] = broker_class

__all__ = ['BrokerManager', 'BrokerFactory', 'PocketOptionBroker']