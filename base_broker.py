# base_broker.py
"""
Base Broker Module

This module defines the abstract base class for broker integrations,
including data models for orders, positions, and account information.

Author: N5StyleEA Team
Version: 15.2.1
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, List, Optional, Any, Callable, Union
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    """Time in force for orders."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Day order

class PositionSide(Enum):
    """Position side (long/short)."""
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    params: Dict[str, Any] = None

    def __post_init__(self):
        """Validate order parameters."""
        if self.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"Price is required for {self.order_type.value} orders")
        
        if self.order_type == OrderType.STOP_LIMIT and self.stop_price is None:
            raise ValueError("Stop price is required for stop-limit orders")

@dataclass
class OrderStatus:
    """Represents the status of an order."""
    order_id: str
    status: str
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    timestamp: Optional[int] = None
    client_order_id: Optional[str] = None
    error: Optional[str] = None

@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    side: PositionSide
    leverage: float = 1.0
    position_id: Optional[str] = None
    timestamp: Optional[int] = None

    @property
    def market_value(self) -> float:
        """Calculate the current market value of the position."""
        return self.quantity * self.current_price

@dataclass
class AccountInfo:
    """Represents account information and balance."""
    equity: float
    balance: float
    margin_available: float
    margin_used: float
    margin_level: Optional[float] = None
    leverage: float = 1.0
    positions: List[Position] = None

    def __post_init__(self):
        """Initialize positions list if None."""
        if self.positions is None:
            self.positions = []

class BaseBroker(ABC):
    """
    Abstract base class for broker integrations.
    All broker implementations should inherit from this class.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        is_testnet: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the broker client.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            is_testnet: Whether to use testnet/sandbox environment
            config: Additional configuration parameters
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connected = False
        self.callbacks = {
            'on_connect': [],
            'on_disconnect': [],
            'on_error': [],
            'on_candle': [],
            'on_tick': [],
            'on_order': [],
            'on_balance': [],
            'on_position': []
        }

    # Connection Management
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the broker's API.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the broker's API.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        pass

    # Account Methods
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Get account information including balance, equity, and open positions.
        
        Returns:
            AccountInfo: Account information object
        """
        pass

    # Order Methods
    @abstractmethod
    async def place_order(self, order: Order) -> OrderStatus:
        """
        Place a new order.
        
        Args:
            order: Order object containing order details
            
        Returns:
            OrderStatus: Status of the placed order
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Optional symbol for the order (required by some brokers)
            
        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str = None) -> OrderStatus:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to check
            symbol: Optional symbol for the order (required by some brokers)
            
        Returns:
            OrderStatus: Current status of the order
        """
        pass

    # Position Methods
    @abstractmethod
    async def get_positions(self, symbol: str = None) -> List[Position]:
        """
        Get current open positions.
        
        Args:
            symbol: Optional symbol to filter positions by
            
        Returns:
            List[Position]: List of open positions
        """
        pass

    @abstractmethod
    async def close_position(self, symbol: str, position_id: str = None) -> bool:
        """
        Close an open position.
        
        Args:
            symbol: Symbol of the position to close
            position_id: Optional position ID (if None, closes all positions for the symbol)
            
        Returns:
            bool: True if position was closed successfully, False otherwise
        """
        pass

    # Market Data Methods
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for candles (e.g., '1m', '1h', '1d')
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Maximum number of candles to return
            
        Returns:
            List[Dict[str, Any]]: List of candle data
        """
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """
        Get the current market price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Current market price
        """
        pass

    # Callback Management
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback function for broker events.
        
        Args:
            event: Event name (e.g., 'on_tick', 'on_order')
            callback: Callback function to register
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")

    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """
        Trigger registered callbacks for an event.
        
        Args:
            event: Event name
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
        """
        if event not in self.callbacks:
            self.logger.warning(f"No callbacks registered for event: {event}")
            return

        for callback in self.callbacks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {event} callback: {str(e)}", exc_info=True)

    # Utility Methods
    async def reconnect(self) -> bool:
        """Reconnect to the broker's API."""
        await self.disconnect()
        return await self.connect()

    def set_demo_mode(self, demo: bool = True) -> None:
        """Switch between demo and live trading."""
        self.is_testnet = demo
        self.logger.info(f"Switched to {'demo' if demo else 'live'} mode")

    async def subscribe_to_symbol(self, symbol: str, timeframe: str) -> bool:
        """
        Subscribe to market data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data updates
            
        Returns:
            bool: True if subscription was successful
        """
        raise NotImplementedError("Symbol subscription not implemented for this broker")

    # Context Manager Support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()