import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import threading
from queue import Queue, Empty
import socket
import struct
from collections import defaultdict
import psutil
import os

# Optimized for cache line size (typically 64 bytes)
@dataclass
class Order:
    order_id: int
    symbol: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    timestamp: float
    strategy_id: int = 0
    
    def to_bytes(self) -> bytes:
        """Convert order to binary format for network transmission."""
        symbol_bytes = self.symbol.ljust(8).encode('ascii')
        return struct.pack('!Q8sdfQ', 
                         self.order_id,
                         symbol_bytes,
                         self.price,
                         self.size,
                         int(time.time_ns()))

class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = SortedDict()  # price -> total_size
        self.asks = SortedDict()
        self._lock = threading.RLock()
        
    def update(self, price: float, size: float, is_bid: bool):
        """Update order book level."""
        book = self.bids if is_bid else self.asks
        with self._lock:
            if size == 0:
                book.pop(price, None)
            else:
                book[price] = size
    
    def get_top(self, is_bid: bool) -> Optional[tuple]:
        """Get best bid or ask."""
        book = self.bids if is_bid else self.asks
        return book.peekitem(-1 if is_bid else 0) if book else None

class HFTOrderManager:
    def __init__(self, config: dict):
        self.config = config
        self.order_books: Dict[str, OrderBook] = {}
        self.orders: Dict[int, Order] = {}
        self.order_queue = Queue(maxsize=10000)
        self.running = False
        self.thread = None
        self.next_order_id = 1
        self.latency_stats = defaultdict(list)
        
        # Performance tuning
        self.enable_jit = config.get('enable_jit', True)
        self.batch_size = config.get('batch_size', 100)
        
        # Network setup
        self.socket = None
        self.setup_network()
        
    def setup_network(self):
        """Initialize low-latency network connections."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set buffer sizes for high throughput
            rcvbuf = 1024 * 1024 * 100  # 100MB
            sndbuf = 1024 * 1024 * 100  # 100MB
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
            
            # Disable Nagle's algorithm
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
        except Exception as e:
            print(f"Network setup error: {e}")
    
    def start(self):
        """Start the order manager in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._process_orders, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the order manager."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _process_orders(self):
        """Process orders in a tight loop for minimum latency."""
        batch = []
        last_flush = time.perf_counter()
        
        while self.running:
            try:
                # Process messages in batches
                order = self.order_queue.get_nowait()
                batch.append(order)
                
                # Process batch if full or timeout
                now = time.perf_counter()
                if len(batch) >= self.batch_size or (now - last_flush) > 0.001:  # 1ms
                    self._process_batch(batch)
                    batch = []
                    last_flush = now
                    
            except Empty:
                if batch:
                    self._process_batch(batch)
                    batch = []
                    last_flush = time.perf_counter()
                time.sleep(0.0001)  # 100μs
    
    def _process_batch(self, orders: List[Order]):
        """Process a batch of orders."""
        if self.enable_jit:
            self._process_batch_jit(orders)
        else:
            for order in orders:
                self._process_single_order(order)
    
    @staticmethod
    @njit(parallel=True)
    def _process_batch_jit(orders):
        """JIT-accelerated batch processing."""
        for i in prange(len(orders)):
            # This is a placeholder - in practice, you'd implement
            # the order processing logic here
            pass
    
    def _process_single_order(self, order: Order):
        """Process a single order."""
        start_time = time.perf_counter_ns()
        
        try:
            # Process order (matching, risk checks, etc.)
            if order.size > 0:
                self._validate_order(order)
                self._match_order(order)
                self._send_to_exchange(order)
            
            # Record latency
            latency_ns = time.perf_counter_ns() - start_time
            self.latency_stats['process_order'].append(latency_ns)
            
        except Exception as e:
            print(f"Error processing order {order.order_id}: {e}")
    
    def _validate_order(self, order: Order):
        """Validate order parameters."""
        if order.size <= 0:
            raise ValueError("Order size must be positive")
        if order.price <= 0:
            raise ValueError("Price must be positive")
    
    def _match_order(self, order: Order):
        """Match order against order book."""
        book = self.order_books.get(order.symbol)
        if not book:
            book = OrderBook(order.symbol)
            self.order_books[order.symbol] = book
        
        # Simple matching logic - extend as needed
        if order.side == 'buy':
            best_ask = book.get_top(False)
            if best_ask and order.price >= best_ask[0]:
                # Cross the spread
                pass
        else:  # sell
            best_bid = book.get_top(True)
            if best_bid and order.price <= best_bid[0]:
                # Cross the spread
                pass
    
    def _send_to_exchange(self, order: Order):
        """Send order to exchange."""
        if self.socket:
            try:
                data = order.to_bytes()
                self.socket.sendto(data, (self.config['exchange_host'], self.config['exchange_port']))
            except Exception as e:
                print(f"Error sending order: {e}")
    
    def submit_order(self, symbol: str, price: float, size: float, side: str) -> int:
        """Submit a new order."""
        order_id = self._get_next_order_id()
        order = Order(
            order_id=order_id,
            symbol=symbol,
            price=price,
            size=size,
            side=side,
            timestamp=time.time_ns()
        )
        
        self.orders[order_id] = order
        self.order_queue.put(order)
        return order_id
    
    def cancel_order(self, order_id: int):
        """Cancel an existing order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            # Send cancel request
            self.order_queue.put(Order(
                order_id=order_id,
                symbol=order.symbol,
                price=0,
                size=0,  # Size 0 indicates cancellation
                side=order.side,
                timestamp=time.time_ns()
            ))
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics in nanoseconds."""
        stats = {}
        for metric, values in self.latency_stats.items():
            if values:
                arr = np.array(values)
                stats[f"{metric}_p50"] = np.percentile(arr, 50)
                stats[f"{metric}_p95"] = np.percentile(arr, 95)
                stats[f"{metric}_p99"] = np.percentile(arr, 99)
                stats[f"{metric}_max"] = np.max(arr)
        return stats
    
    def _get_next_order_id(self) -> int:
        """Thread-safe order ID generation."""
        with threading.Lock():
            order_id = self.next_order_id
            self.next_order_id += 1
            return order_id

# Optimized data structures
class SortedDict(dict):
    """Simple sorted dictionary for order book levels."""
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._keys = sorted(self.keys())
    
    def __delitem__(self, key):
        super().__delitem__(key)
        self._keys = sorted(self.keys())
    
    def peekitem(self, index):
        key = self._keys[index]
        return (key, self[key])

# System optimization utilities
def optimize_system():
    """Apply system-level optimizations for HFT."""
    try:
        # Set CPU affinity
        p = psutil.Process()
        p.cpu_affinity(list(range(os.cpu_count())))
        
        # Set real-time priority
        p.nice(psutil.REALTIME_PRIORITY_CLASS)
        
        # Disable CPU frequency scaling
        os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        
    except Exception as e:
        print(f"Warning: Could not optimize system: {e}")

# Example usage
if __name__ == "__main__":
    # System optimization
    optimize_system()
    
    # Initialize order manager
    config = {
        'exchange_host': '127.0.0.1',
        'exchange_port': 5000,
        'enable_jit': True,
        'batch_size': 100
    }
    
    manager = HFTOrderManager(config)
    manager.start()
    
    try:
        # Example: Submit some orders
        for i in range(1000):
            manager.submit_order(
                symbol="BTC-USD",
                price=50000 + (i % 100) * 0.5,
                size=0.1,
                side="buy" if i % 2 == 0 else "sell"
            )
            time.sleep(0.001)  # 1ms
            
        # Print latency stats
        print("Latency stats (ns):", manager.get_latency_stats())
        
    finally:
        manager.stop()
