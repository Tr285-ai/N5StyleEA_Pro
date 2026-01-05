import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from queue import Queue, PriorityQueue

@dataclass
class IcebergOrder:
    """Implementation of Iceberg Order (Large order divided into smaller visible parts)"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    total_quantity: float
    display_quantity: float
    price: float
    order_type: str = "LIMIT"
    time_in_force: str = "GTC"
    
    def __post_init__(self):
        self.remaining_quantity = self.total_quantity
        self.visible_quantity = min(self.display_quantity, self.remaining_quantity)
        self.status = "NEW"
        self.created_at = datetime.utcnow()
        self.last_updated = self.created_at
    
    def get_visible_order(self) -> dict:
        """Get the visible portion of the iceberg order"""
        return {
            'order_id': f"{self.order_id}_{int(time.time()*1000)}",
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.visible_quantity,
            'price': self.price,
            'order_type': self.order_type,
            'time_in_force': self.time_in_force,
            'iceberg_id': self.order_id
        }
    
    def update_after_fill(self, filled_quantity: float) -> bool:
        """Update order after a fill and return if there's more to fill"""
        self.remaining_quantity -= filled_quantity
        self.last_updated = datetime.utcnow()
        
        if self.remaining_quantity <= 0:
            self.status = "FILLED"
            return False
            
        # Prepare next slice
        self.visible_quantity = min(self.display_quantity, self.remaining_quantity)
        return True

@dataclass
class TWAPOrder:
    """Time-Weighted Average Price Order"""
    order_id: str
    symbol: str
    side: str
    total_quantity: float
    duration_seconds: int
    price_limit: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        self.start_time = self.start_time or datetime.utcnow()
        self.end_time = self.end_time or (self.start_time + timedelta(seconds=self.duration_seconds))
        self.remaining_quantity = self.total_quantity
        self.status = "NEW"
        self.executions = []
        self.next_slice_time = self.start_time
        self.slice_duration = self.duration_seconds / max(1, (self.total_quantity // 0.1))  # At least 0.1 per slice
    
    def get_next_slice(self, current_time: datetime) -> Optional[dict]:
        """Get the next slice of the TWAP order"""
        if current_time < self.start_time or current_time >= self.end_time:
            return None
            
        if current_time >= self.next_slice_time and self.remaining_quantity > 0:
            # Calculate slice size based on time remaining
            time_elapsed = (current_time - self.start_time).total_seconds()
            time_remaining = max(1, (self.end_time - current_time).total_seconds())
            target_slice = (self.total_quantity * (time_elapsed / self.duration_seconds)) - sum(e['quantity'] for e in self.executions)
            slice_size = max(0.01, min(self.remaining_quantity, target_slice))
            
            if slice_size <= 0:
                self.next_slice_time = current_time + timedelta(seconds=self.slice_duration)
                return None
                
            self.next_slice_time = current_time + timedelta(seconds=self.slice_duration)
            return {
                'order_id': f"{self.order_id}_{int(current_time.timestamp()*1000)}",
                'symbol': self.symbol,
                'side': self.side,
                'quantity': slice_size,
                'order_type': 'LIMIT' if self.price_limit else 'MARKET',
                'price': self.price_limit,
                'twap_id': self.order_id
            }
        return None
    
    def update_after_fill(self, fill_quantity: float, avg_price: float):
        """Update order after a fill"""
        self.remaining_quantity -= fill_quantity
        self.executions.append({
            'timestamp': datetime.utcnow(),
            'quantity': fill_quantity,
            'price': avg_price
        })
        
        if self.remaining_quantity <= 0:
            self.status = "FILLED"
        elif self.status == "NEW":
            self.status = "PARTIALLY_FILLED"

class DarkPoolRouter:
    """Dark Pool Integration"""
    def __init__(self, dark_pools: List[dict]):
        self.dark_pools = dark_pools  # List of {'name': str, 'fee': float, 'liquidity': float}
        self.available_pools = self._discover_pools()
        
    def _discover_pools(self) -> List[dict]:
        """Discover available dark pools"""
        # In a real implementation, this would query dark pool APIs
        return [pool for pool in self.dark_pools if self._check_pool_availability(pool)]
    
    def _check_pool_availability(self, pool: dict) -> bool:
        """Check if a dark pool is available"""
        # Implementation would check pool status, connectivity, etc.
        return True
    
    def find_best_pool(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        """Find the best dark pool for the given order"""
        if not self.available_pools:
            return None
            
        # Simple implementation: choose pool with highest liquidity
        # In practice, this would consider fees, fill probability, etc.
        return max(self.available_pools, key=lambda x: x.get('liquidity', 0))
    
    def route_to_dark_pool(self, order: dict) -> dict:
        """Route order to dark pool"""
        pool = self.find_best_pool(order['symbol'], order['side'], order['quantity'])
        if not pool:
            return {'success': False, 'error': 'No dark pools available'}
            
        # In a real implementation, this would make an API call to the dark pool
        return {
            'success': True,
            'pool': pool['name'],
            'order_id': f"DP_{order['order_id']}",
            'timestamp': datetime.utcnow().isoformat()
        }

class AdvancedOrderManager:
    """Manages advanced order types"""
    def __init__(self, dark_pools: List[dict] = None):
        self.iceberg_orders: Dict[str, IcebergOrder] = {}
        self.twap_orders: Dict[str, TWAPOrder] = {}
        self.dark_pool = DarkPoolRouter(dark_pools or [])
        self.order_queue = PriorityQueue()
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """Start the order manager"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_orders, daemon=True)
        self.worker_thread.start()
        
    def stop(self):
        """Stop the order manager"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def create_iceberg_order(self, order_id: str, symbol: str, side: str, 
                           total_quantity: float, display_quantity: float, 
                           price: float, order_type: str = "LIMIT") -> dict:
        """Create a new iceberg order"""
        if order_id in self.iceberg_orders:
            return {'success': False, 'error': 'Order ID already exists'}
            
        order = IcebergOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            display_quantity=display_quantity,
            price=price,
            order_type=order_type
        )
        self.iceberg_orders[order_id] = order
        self.order_queue.put((1, order))  # Higher priority for new orders
        return {'success': True, 'order_id': order_id}
    
    def create_twap_order(self, order_id: str, symbol: str, side: str, 
                         total_quantity: float, duration_seconds: int,
                         price_limit: float = None) -> dict:
        """Create a new TWAP order"""
        if order_id in self.twap_orders:
            return {'success': False, 'error': 'Order ID already exists'}
            
        order = TWAPOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            duration_seconds=duration_seconds,
            price_limit=price_limit
        )
        self.twap_orders[order_id] = order
        self.order_queue.put((1, order))  # Higher priority for new orders
        return {'success': True, 'order_id': order_id}
    
    def route_to_dark_pool(self, order: dict) -> dict:
        """Route an order to a dark pool"""
        return self.dark_pool.route_to_dark_pool(order)
    
    def _process_orders(self):
        """Main order processing loop"""
        while self.running:
            try:
                _, order = self.order_queue.get(timeout=0.1)
                self._process_order(order)
            except Exception as e:
                print(f"Error processing order: {e}")
                continue
    
    def _process_order(self, order):
        """Process a single order"""
        try:
            if isinstance(order, IcebergOrder):
                self._process_iceberg_order(order)
            elif isinstance(order, TWAPOrder):
                self._process_twap_order(order)
        except Exception as e:
            print(f"Error processing {type(order).__name__} {order.order_id}: {e}")
    
    def _process_iceberg_order(self, order: IcebergOrder):
        """Process an iceberg order"""
        if order.status == "FILLED":
            return
            
        # Get visible slice
        visible_order = order.get_visible_order()
        
        # In a real implementation, this would submit to the exchange
        print(f"Submitting visible slice: {visible_order}")
        
        # Simulate fill (in real code, this would be a callback from the exchange)
        filled_qty = min(visible_order['quantity'], order.remaining_quantity * 0.8)  # Simulate partial fill
        if filled_qty > 0:
            if order.update_after_fill(filled_qty):
                # Resubmit for next slice
                self.order_queue.put((0, order))  # Lower priority for follow-up slices
    
    def _process_twap_order(self, order: TWAPOrder):
        """Process a TWAP order"""
        current_time = datetime.utcnow()
        slice_order = order.get_next_slice(current_time)
        
        if slice_order:
            # In a real implementation, this would submit to the exchange
            print(f"Submitting TWAP slice: {slice_order}")
            
            # Simulate fill (in real code, this would be a callback from the exchange)
            filled_qty = slice_order['quantity'] * 0.9  # Simulate 90% fill
            order.update_after_fill(filled_qty, slice_order.get('price', 0))
            
            if order.remaining_quantity > 0 and current_time < order.end_time:
                # Reschedule for next slice
                self.order_queue.put((0, order))  # Lower priority for follow-up slices

# Example usage
if __name__ == "__main__":
    # Initialize with some dark pools
    dark_pools = [
        {'name': 'Liquidnet', 'fee': 0.0005, 'liquidity': 1000000},
        {'name': 'ITG POSIT', 'fee': 0.0003, 'liquidity': 750000},
        {'name': 'UBS ATS', 'fee': 0.0004, 'liquidity': 500000}
    ]
    
    manager = AdvancedOrderManager(dark_pools)
    manager.start()
    
    try:
        # Example: Create an iceberg order
        manager.create_iceberg_order(
            order_id="ICEBERG_001",
            symbol="AAPL",
            side="BUY",
            total_quantity=1000,
            display_quantity=100,
            price=150.0
        )
        
        # Example: Create a TWAP order
        manager.create_twap_order(
            order_id="TWAP_001",
            symbol="MSFT",
            side="SELL",
            total_quantity=2000,
            duration_seconds=3600,  # 1 hour
            price_limit=300.0
        )
        
        # Example: Route to dark pool
        dark_pool_order = {
            'order_id': 'DARK_001',
            'symbol': 'GOOG',
            'side': 'BUY',
            'quantity': 500,
            'price': 2800.0
        }
        result = manager.route_to_dark_pool(dark_pool_order)
        print(f"Dark pool routing result: {result}")
        
        # Keep the example running for a while
        import time
        time.sleep(5)
        
    finally:
        manager.stop()
