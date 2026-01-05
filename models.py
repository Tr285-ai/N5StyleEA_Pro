from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Day order

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
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'client_order_id': self.client_order_id,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'metadata': self.metadata
        }

@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: OrderSide
    unrealized_pnl: float
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin_used: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'side': self.side.value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'leverage': self.leverage,
            'margin_used': self.margin_used,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class Account:
    """Represents a trading account."""
    account_id: str
    balance: float
    equity: float
    margin_available: float
    margin_used: float
    open_positions: List[Position] = field(default_factory=list)
    open_orders: List[Order] = field(default_factory=list)
    leverage: float = 1.0
    is_live: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert account to dictionary."""
        return {
            'account_id': self.account_id,
            'balance': self.balance,
            'equity': self.equity,
            'margin_available': self.margin_available,
            'margin_used': self.margin_used,
            'leverage': self.leverage,
            'is_live': self.is_live,
            'open_positions': [pos.to_dict() for pos in self.open_positions],
            'open_orders': [order.to_dict() for order in self.open_orders],
            'metadata': self.metadata
        }

class Trade:
    """Trade model for storing trade information."""
    
    STATUS_PENDING = 'pending'
    STATUS_APPROVED = 'approved'
    STATUS_REJECTED = 'rejected'
    STATUS_CLOSED = 'closed'
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.user_id = kwargs.get('user_id')
        self.symbol = kwargs.get('symbol')
        self.direction = kwargs.get('direction')
        self.amount = kwargs.get('amount')
        self.price = kwargs.get('price')
        self.status = kwargs.get('status', self.STATUS_PENDING)
        self.created_at = kwargs.get('created_at')
        self.updated_at = kwargs.get('updated_at')
        self.closed_at = kwargs.get('closed_at')
        self.close_price = kwargs.get('close_price')
        self.pnl = kwargs.get('pnl')
        self.pnl_percentage = kwargs.get('pnl_percentage')
    