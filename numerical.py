# optimizations/numerical.py
import numpy as np
from numba import jit, njit, prange
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from numba.typed import List as NumbaList
import time
import logging

logger = logging.getLogger(__name__)

class NumericalOptimizer:
    """Handles numerical computations with Numba JIT compilation."""
    
    def __init__(self, enable_parallel: bool = True):
        """
        Initialize the numerical optimizer.
        
        Args:
            enable_parallel: Whether to enable parallel processing
        """
        self.enable_parallel = enable_parallel
        self._compiled_functions = {}
        
    def _get_cache_key(self, func_name: str, *args) -> str:
        """Generate a cache key for function compilation."""
        arg_types = tuple(type(arg).__name__ for arg in args)
        return f"{func_name}_{'_'.join(arg_types)}"
    
    def optimize_dataframe_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame operations using vectorized operations and JIT.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Optimized DataFrame
        """
        # Convert to numpy arrays for faster processing
        close_prices = df['close'].values
        volumes = df['volume'].values
        
        # Apply JIT-compiled functions
        returns = self.calculate_returns(close_prices)
        volatility = self.calculate_volatility(returns, window=20)
        vwap = self.calculate_vwap(close_prices, volumes)
        
        # Add calculated values to DataFrame
        df['returns'] = returns
        df['volatility'] = volatility
        df['vwap'] = vwap
        
        return df
    
    @staticmethod
    @njit(fastmath=True, parallel=False)
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate percentage returns with Numba JIT."""
        returns = np.zeros_like(prices)
        for i in prange(1, len(prices)):
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1] * 100
        return returns
    
    @staticmethod
    @njit(fastmath=True)
    def calculate_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate rolling volatility with Numba JIT."""
        volatility = np.zeros_like(returns)
        for i in prange(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i]) * np.sqrt(252)  # Annualized
        return volatility
    
    @staticmethod
    @njit(fastmath=True)
    def calculate_vwap(prices: np.ndarray, volumes: np.ndarray, 
                      window: int = 20) -> np.ndarray:
        """Calculate Volume Weighted Average Price with Numba JIT."""
        vwap = np.zeros_like(prices)
        for i in prange(window, len(prices)):
            window_prices = prices[i-window:i]
            window_volumes = volumes[i-window:i]
            vwap[i] = np.sum(window_prices * window_volumes) / np.sum(window_volumes)
        return vwap