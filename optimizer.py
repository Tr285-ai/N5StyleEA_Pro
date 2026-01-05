# optim/optimizer.py
import time
import cProfile
import pstats
import io
import tracemalloc
from functools import wraps
from typing import Callable, Any, Dict, List
import pandas as pd
import numpy as np

class PerformanceOptimizer:
    """Utility class for performance optimization."""
    
    @staticmethod
    def profile(func: Callable) -> Callable:
        """Decorator to profile function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()
            
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            print(s.getvalue())
            
            return result
        return wrapper
    
    @staticmethod
    def memory_usage(func: Callable) -> Callable:
        """Decorator to measure memory usage of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"Memory usage - Current: {current / 10**6:.2f}MB, Peak: {peak / 10**6:.2f}MB")
            return result
        return wrapper
    
    @staticmethod
    def time_execution(func: Callable) -> Callable:
        """Decorator to measure execution time of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
            return result
        return wrapper

class DataOptimizer:
    """Optimize data processing operations."""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        df_optimized = df.copy()
        
        # Downcast numeric columns
        for col in df_optimized.select_dtypes(include=['int']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            
        for col in df_optimized.select_dtypes(include=['float']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            
        # Convert object types to category if low cardinality
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
                
        return df_optimized
    
    @staticmethod
    def batch_process(
        data: List[Any],
        process_func: Callable,
        batch_size: int = 1000
    ) -> List[Any]:
        """Process data in batches to reduce memory usage."""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            results.extend(process_func(batch))
        return results