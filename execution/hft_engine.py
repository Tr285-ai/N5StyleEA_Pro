import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time
import socket
import struct
import threading
from queue import Queue, Empty
import numba

class HFTOrder:
    def __init__(self, order_id: str, symbol: str, side: str, price: float, size: float, 
                 order_type: str = "LIMIT", time_in_force: str = "IOC"):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side.upper()
        self.price = price
        self.size = size
        self.order_type = order_type
        self.time_in_force = time_in_force
        self.timestamp = time.time_ns()
        self.status = "NEW"
        
    def __repr__(self):
        return f"{self.symbol} {self.side} {self.size}@{self.price} ({self.status})"

class HFTExecutionEngine:
    """High-Frequency Trading Execution Engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.order_book = {}
        self.order_queue = Queue()
        self.market_data = {}
        self.running = False
        self.latency_optimized = config.get('latency_optimized', True)
        self.use_udp = config.get('use_udp', True)
        
        # Performance metrics
        self.metrics = {
            'orders_processed': 0,
            'avg_latency_ns': 0,
            'total_volume': 0.0,
            'rejected_orders': 0
        }
        
        # Initialize network
        self._init_network()
        
    def _init_network(self):
        """Initialize low-latency network connections"""
        if self.use_udp:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Enable low-latency options if on Linux
            try:
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)  # Higher priority
            except:
                pass
    
    def start(self):
        """Start the execution engine"""
        self.running = True
        self.execution_thread = threading.Thread(target=self._process_orders, daemon=True)
        self.execution_thread.start()
        
    def stop(self):
        """Stop the execution engine"""
        self.running = False
        if hasattr(self, 'execution_thread'):
            self.execution_thread.join()
    
    def submit_order(self, order: HFTOrder) -> bool:
        """Submit an order for execution"""
        order.timestamp = time.time_ns()
        self.order_queue.put(order)
        self.order_book[order.order_id] = order
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id in self.order_book:
            self.order_book[order_id].status = "CANCELLED"
            return True
        return False
    
    def _process_orders(self):
        """Main order processing loop"""
        while self.running:
            try:
                order = self.order_queue.get(timeout=0.001)  # Non-blocking get
                self._execute_order(order)
            except Empty:
                continue
    
    @numba.jit(nopython=True)
    def _execute_order(self, order: HFTOrder):
        """Execute a single order with low-latency optimizations"""
        start_time = time.time_ns()
        
        try:
            # Simulate order matching and execution
            if order.status == "NEW":
                order.status = "FILLED"
                self.metrics['orders_processed'] += 1
                self.metrics['total_volume'] += order.size
                
                # Calculate latency
                latency = time.time_ns() - order.timestamp
                # Update running average
                self.metrics['avg_latency_ns'] = (
                    (self.metrics['avg_latency_ns'] * (self.metrics['orders_processed'] - 1) + latency) / 
                    self.metrics['orders_processed']
                )
                
        except Exception as e:
            order.status = "REJECTED"
            self.metrics['rejected_orders'] += 1
            
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.metrics
        
    def optimize_network_path(self):
        """Optimize network path for lowest latency"""
        # Implementation for network optimization
        pass
        
    def pre_warm_cache(self):
        """Pre-warm CPU cache for critical code paths"""
        # Implementation for cache optimization
        pass

# Example usage
if __name__ == "__main__":
    config = {
        'latency_optimized': True,
        'use_udp': True,
        'max_orders_per_second': 100000
    }
    
    engine = HFTExecutionEngine(config)
    engine.start()
    
    # Example order
    order = HFTOrder(
        order_id="ORD12345",
        symbol="BTC/USDT",
        side="BUY",
        price=50000.0,
        size=0.1
    )
    
    engine.submit_order(order)
    time.sleep(0.1)  # Give time for processing
    print(f"Order status: {order.status}")
    print(f"Metrics: {engine.get_performance_metrics()}")
    
    engine.stop()
