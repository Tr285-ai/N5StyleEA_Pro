# exchanges/exchange_factory.py
from typing import Dict, Type
from abc import ABC, abstractmethod
import ccxt
import logging

logger = logging.getLogger(__name__)

class BaseExchange(ABC):
    """Base class for all exchange implementations."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the exchange."""
        pass
        
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass
        
    @abstractmethod
    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> dict:
        """Place a new order."""
        pass

class BinanceExchange(BaseExchange):
    """Binance exchange implementation."""
    
    def connect(self) -> bool:
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False

class FTXExchange(BaseExchange):
    """FTX exchange implementation."""
    
    def connect(self) -> bool:
        try:
            self.exchange = ccxt.ftx({
                'apiKey': self.api_key,
                'secret': self.api_secret
            })
            return True
        except Exception as e:
            logger.error(f"Failed to connect to FTX: {e}")
            return False

class ExchangeFactory:
    """Factory class for creating exchange instances."""
    
    _exchanges = {
        'binance': BinanceExchange,
        'ftx': FTXExchange,
        # Add more exchanges here
    }
    
    @classmethod
    def get_exchange(cls, exchange_name: str, api_key: str = None, api_secret: str = None) -> BaseExchange:
        """Get an exchange instance by name."""
        exchange_class = cls._exchanges.get(exchange_name.lower())
        if not exchange_class:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        return exchange_class(api_key, api_secret)
    
    @classmethod
    def register_exchange(cls, name: str, exchange_class: Type[BaseExchange]):
        """Register a new exchange implementation."""
        if not issubclass(exchange_class, BaseExchange):
            raise TypeError("Exchange class must inherit from BaseExchange")
        cls._exchanges[name.lower()] = exchange_class