""
Trading System Compliance Manager

This module handles compliance requirements including audit trails,
order tagging, and regulatory reporting.
"""
import os
import json
import csv
import time
import hashlib
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import threading
import queue
from pathlib import Path
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compliance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class Order:
    """Represents a trading order with compliance metadata."""
    order_id: str
    client_order_id: str
    symbol: str
    type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK, etc.
    status: OrderStatus = OrderStatus.NEW
    timestamp: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'type': self.type.value,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price is not None else None,
            'stop_price': str(self.stop_price) if self.stop_price is not None else None,
            'time_in_force': self.time_in_force,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'filled_quantity': str(self.filled_quantity),
            'avg_fill_price': str(self.avg_fill_price) if self.avg_fill_price is not None else None,
            'tags': self.tags,
            'metadata': self.metadata
        }

@dataclass
class Trade:
    """Represents a trade execution."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    commission_asset: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_maker: bool = False
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'price': str(self.price),
            'commission': str(self.commission),
            'commission_asset': self.commission_asset,
            'timestamp': self.timestamp.isoformat(),
            'is_maker': self.is_maker,
            'tags': self.tags,
            'metadata': self.metadata
        }

class ComplianceManager:
    """Manages compliance requirements for the trading system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the compliance manager."""
        self.config = config or {}
        self.orders: Dict[str, Order] = {}
        self.trades: Dict[str, Trade] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.running = False
        self.thread = None
        self.queue = queue.Queue()
        self.data_dir = Path(self.config.get('data_dir', 'data/compliance'))
        self.retention_days = self.config.get('retention_days', 365 * 5)  # 5 years by default
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backends
        self.storage_backends = self._init_storage_backends()
    
    def _init_storage_backends(self) -> List[Callable[[Dict[str, Any]], None]]:
        """Initialize storage backends for audit logs."""
        backends = []
        
        # Local file storage
        backends.append(self._store_local)
        
        # Add database storage if configured
        if self.config.get('database'):
            backends.append(self._store_database)
        
        # Add cloud storage if configured
        if self.config.get('cloud_storage'):
            backends.append(self._store_cloud)
        
        return backends
    
    def start(self):
        """Start the compliance manager."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_queue, daemon=True)
            self.thread.start()
            logger.info("Compliance manager started")
    
    def stop(self):
        """Stop the compliance manager."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _process_queue(self):
        """Process events from the queue."""
        while self.running:
            try:
                event = self.queue.get(timeout=1)
                self._handle_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing compliance event: {e}", exc_info=True)
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle a compliance event."""
        try:
            event_type = event.get('type')
            
            if event_type == 'order':
                self._handle_order_event(event)
            elif event_type == 'trade':
                self._handle_trade_event(event)
            elif event_type == 'system':
                self._handle_system_event(event)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                return
            
            # Add to audit log
            self._add_to_audit_log(event)
            
            # Store in all backends
            for backend in self.storage_backends:
                try:
                    backend(event)
                except Exception as e:
                    logger.error(f"Error storing event in backend: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling compliance event: {e}", exc_info=True)
    
    def _handle_order_event(self, event: Dict[str, Any]):
        """Handle an order event."""
        order_data = event['data']
        order_id = order_data['order_id']
        
        if order_id in self.orders:
            # Update existing order
            order = self.orders[order_id]
            for key, value in order_data.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            order.last_updated = datetime.utcnow()
        else:
            # Create new order
            order = Order(
                order_id=order_data['order_id'],
                client_order_id=order_data.get('client_order_id', str(uuid.uuid4())),
                symbol=order_data['symbol'],
                type=OrderType(order_data['type']),
                side=OrderSide(order_data['side']),
                quantity=float(order_data['quantity']),
                price=float(order_data['price']) if order_data.get('price') else None,
                stop_price=float(order_data['stop_price']) if order_data.get('stop_price') else None,
                time_in_force=order_data.get('time_in_force', 'GTC'),
                status=OrderStatus(order_data.get('status', 'NEW')),
                timestamp=datetime.fromisoformat(order_data.get('timestamp')) if order_data.get('timestamp') else datetime.utcnow(),
                tags=order_data.get('tags', {}),
                metadata=order_data.get('metadata', {})
            )
            self.orders[order_id] = order
    
    def _handle_trade_event(self, event: Dict[str, Any]):
        """Handle a trade event."""
        trade_data = event['data']
        trade_id = trade_data['trade_id']
        
        if trade_id not in self.trades:
            # Create new trade
            trade = Trade(
                trade_id=trade_id,
                order_id=trade_data['order_id'],
                symbol=trade_data['symbol'],
                side=OrderSide(trade_data['side']),
                quantity=float(trade_data['quantity']),
                price=float(trade_data['price']),
                commission=float(trade_data.get('commission', 0.0)),
                commission_asset=trade_data.get('commission_asset', 'USDT'),
                timestamp=datetime.fromisoformat(trade_data.get('timestamp')) if trade_data.get('timestamp') else datetime.utcnow(),
                is_maker=trade_data.get('is_maker', False),
                tags=trade_data.get('tags', {}),
                metadata=trade_data.get('metadata', {})
            )
            self.trades[trade_id] = trade
    
    def _handle_system_event(self, event: Dict[str, Any]):
        """Handle a system event."""
        # System events are just logged, no special handling needed
        pass
    
    def _add_to_audit_log(self, event: Dict[str, Any]):
        """Add an event to the in-memory audit log."""
        # Add a unique ID and timestamp if not present
        event.setdefault('event_id', str(uuid.uuid4()))
        event.setdefault('timestamp', datetime.utcnow().isoformat())
        
        # Add to audit log
        self.audit_log.append(event)
        
        # Keep only recent events in memory
        max_audit_log_size = self.config.get('max_audit_log_size', 10000)
        if len(self.audit_log) > max_audit_log_size:
            self.audit_log = self.audit_log[-max_audit_log_size:]
    
    def _store_local(self, event: Dict[str, Any]):
        """Store event in local files."""
        # Create daily log files
        timestamp = datetime.fromisoformat(event['timestamp']) if isinstance(event['timestamp'], str) else event['timestamp']
        date_str = timestamp.strftime('%Y-%m-%d')
        
        # Create directory for the date if it doesn't exist
        date_dir = self.data_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        # Write to JSONL file
        log_file = date_dir / f"{event['type']}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def _store_database(self, event: Dict[str, Any]):
        """Store event in database."""
        # Implementation depends on the database backend
        # This is a placeholder for the actual implementation
        pass
    
    def _store_cloud(self, event: Dict[str, Any]):
        """Store event in cloud storage."""
        # Implementation depends on the cloud provider
        # This is a placeholder for the actual implementation
        pass
    
    def record_order(self, order_data: Dict[str, Any]):
        """Record an order event."""
        self.queue.put({
            'type': 'order',
            'timestamp': datetime.utcnow().isoformat(),
            'data': order_data
        })
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a trade event."""
        self.queue.put({
            'type': 'trade',
            'timestamp': datetime.utcnow().isoformat(),
            'data': trade_data
        })
    
    def record_system_event(self, event_type: str, data: Dict[str, Any]):
        """Record a system event."""
        self.queue.put({
            'type': 'system',
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        })
    
    def generate_report(self, report_type: str, start_date: datetime, end_date: datetime) -> str:
        """Generate a compliance report."""
        if report_type == 'trades':
            return self._generate_trades_report(start_date, end_date)
        elif report_type == 'orders':
            return self._generate_orders_report(start_date, end_date)
        elif report_type == 'audit':
            return self._generate_audit_report(start_date, end_date)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def _generate_trades_report(self, start_date: datetime, end_date: datetime) -> str:
        """Generate a trades report in CSV format."""
        output = []
        
        # Add header
        output.append([
            'Trade ID', 'Order ID', 'Symbol', 'Side', 'Quantity', 'Price',
            'Commission', 'Commission Asset', 'Timestamp', 'Is Maker'
        ])
        
        # Add trades
        for trade in self.trades.values():
            if start_date <= trade.timestamp <= end_date:
                output.append([
                    trade.trade_id,
                    trade.order_id,
                    trade.symbol,
                    trade.side.value,
                    str(trade.quantity),
                    str(trade.price),
                    str(trade.commission),
                    trade.commission_asset,
                    trade.timestamp.isoformat(),
                    str(trade.is_maker)
                ])
        
        # Convert to CSV
        import io
        output_io = io.StringIO()
        writer = csv.writer(output_io)
        writer.writerows(output)
        return output_io.getvalue()
    
    def _generate_orders_report(self, start_date: datetime, end_date: datetime) -> str:
        """Generate an orders report in CSV format."""
        output = []
        
        # Add header
        output.append([
            'Order ID', 'Client Order ID', 'Symbol', 'Type', 'Side', 'Quantity',
            'Price', 'Stop Price', 'Time In Force', 'Status', 'Timestamp',
            'Filled Quantity', 'Average Fill Price'
        ])
        
        # Add orders
        for order in self.orders.values():
            if start_date <= order.timestamp <= end_date:
                output.append([
                    order.order_id,
                    order.client_order_id,
                    order.symbol,
                    order.type.value,
                    order.side.value,
                    str(order.quantity),
                    str(order.price) if order.price is not None else '',
                    str(order.stop_price) if order.stop_price is not None else '',
                    order.time_in_force,
                    order.status.value,
                    order.timestamp.isoformat(),
                    str(order.filled_quantity),
                    str(order.avg_fill_price) if order.avg_fill_price is not None else ''
                ])
        
        # Convert to CSV
        import io
        output_io = io.StringIO()
        writer = csv.writer(output_io)
        writer.writerows(output)
        return output_io.getvalue()
    
    def _generate_audit_report(self, start_date: datetime, end_date: datetime) -> str:
        """Generate an audit log report in JSON format."""
        filtered_events = [
            event for event in self.audit_log
            if start_date <= datetime.fromisoformat(event['timestamp']) <= end_date
        ]
        return json.dumps(filtered_events, indent=2)

# Example usage
if __name__ == "__main__":
    # Create compliance manager
    compliance = ComplianceManager({
        'data_dir': 'data/compliance',
        'retention_days': 365 * 5  # Keep data for 5 years
    })
    
    # Start the compliance manager
    compliance.start()
    
    try:
        # Example: Record an order
        compliance.record_order({
            'order_id': '12345',
            'symbol': 'BTCUSDT',
            'type': 'LIMIT',
            'side': 'BUY',
            'quantity': '1.0',
            'price': '50000.0',
            'time_in_force': 'GTC',
            'status': 'NEW',
            'tags': {
                'strategy': 'mean_reversion',
                'trader': 'algo_1'
            }
        })
        
        # Example: Record a trade
        compliance.record_trade({
            'trade_id': '67890',
            'order_id': '12345',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': '1.0',
            'price': '50000.0',
            'commission': '10.0',
            'commission_asset': 'USDT',
            'is_maker': False,
            'tags': {
                'exchange': 'binance'
            }
        })
        
        # Example: Record a system event
        compliance.record_system_event('startup', {
            'version': '1.0.0',
            'hostname': 'trading-server-1',
            'ip_address': '192.168.1.100'
        })
        
        # Wait for events to be processed
        import time
        time.sleep(2)
        
        # Generate a report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        report = compliance.generate_report('trades', start_date, end_date)
        print("Trades Report:")
        print(report)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        compliance.stop()
