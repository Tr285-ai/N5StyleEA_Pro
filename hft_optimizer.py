import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numba as nb
from dataclasses import dataclass
import time

@dataclass
class HFTOptimizationParams:
    max_order_book_depth: int = 10
    latency_threshold_ms: float = 0.5
    max_concurrent_orders: int = 100
    order_size_limit: float = 0.1  # Max order size as % of average daily volume
    price_improvement_target: float = 0.0001  # Target price improvement (0.01%)
    max_position_hold_time: float = 300  # seconds

class HFTOptimizer:
    def __init__(self, params: Optional[HFTOptimizationParams] = None):
        self.params = params or HFTOptimizationParams()
        self.latency_metrics = []
        self.order_book_snapshot = {
            'bids': np.zeros((self.params.max_order_book_depth, 2)),
            'asks': np.zeros((self.params.max_order_book_depth, 2))
        }
        
    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _calculate_market_impact(quantity: float, order_book: np.ndarray, is_buy: bool) -> float:
        """Calculate market impact using volume profile."""
        remaining = quantity
        impact = 0.0
        
        for i in range(len(order_book)):
            price, volume = order_book[i]
            if remaining <= 0:
                break
                
            if volume > remaining:
                impact += remaining * price
                remaining = 0
            else:
                impact += volume * price
                remaining -= volume
                
        return impact / quantity if quantity > 0 else 0.0
    
    def update_order_book(self, order_book: Dict[str, np.ndarray]) -> None:
        """Update order book snapshot with new data."""
        for side in ['bids', 'asks']:
            depth = min(len(order_book[side]), self.params.max_order_book_depth)
            self.order_book_snapshot[side][:depth] = order_book[side][:depth]
    
    def optimize_order_execution(self, side: str, quantity: float) -> Dict:
        """Optimize order execution using VWAP and market impact."""
        start_time = time.time()
        
        # Get appropriate order book side
        ob_side = 'bids' if side == 'sell' else 'asks'
        order_book = self.order_book_snapshot[ob_side]
        
        # Calculate market impact
        impact = self._calculate_market_impact(quantity, order_book, side == 'buy')
        
        # Calculate optimal order size and price
        avg_price = np.average(order_book[:, 0], weights=order_book[:, 1])
        optimal_price = avg_price * (1 - self.params.price_improvement_target) if side == 'sell' else \
                       avg_price * (1 + self.params.price_improvement_target)
        
        # Calculate time-weighted order size
        time_weight = min(1.0, (time.time() - start_time) / self.params.latency_threshold_ms)
        order_size = min(quantity, self.params.order_size_limit * time_weight)
        
        return {
            'price': optimal_price,
            'size': order_size,
            'market_impact': impact,
            'timestamp': time.time(),
            'remaining_quantity': quantity - order_size
        }
    
    def measure_latency(self, start_time: float) -> None:
        """Record latency metrics for performance monitoring."""
        latency = (time.time() - start_time) * 1000  # Convert to ms
        self.latency_metrics.append(latency)
        
    def get_latency_stats(self) -> Dict:
        """Get latency statistics in milliseconds."""
        if not self.latency_metrics:
            return {}
            
        latencies = np.array(self.latency_metrics)
        return {
            'avg_latency_ms': float(np.mean(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'max_latency_ms': float(np.max(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'total_measurements': len(latencies)
        }
