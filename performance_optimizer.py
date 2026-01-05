"""
Performance optimization module with profiling, caching, and parallel processing.
"""
import time
import cProfile
import pstats
import io
import functools
from typing import Callable, Any, Dict, List, Optional, TypeVar, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import psutil
import os

logger = logging.getLogger(__name__)
T = TypeVar('T')  # Generic type variable

class PerformanceProfiler:
    """Performance profiling and timing utilities."""
    
    def __init__(self):
        self.timers = {}
        self.profiler = cProfile.Profile()
        
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = time.perf_counter()
        
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time in seconds."""
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' not found")
            
        elapsed = time.perf_counter() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function's performance."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.profiler.enable()
            result = func(*args, **kwargs)
            self.profiler.disable()
            return result
        return wrapper
    
    def get_profile_stats(self, sort_by: str = 'cumulative', limit: int = 20) -> str:
        """Get profiling statistics as a formatted string."""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(limit)
        return s.getvalue()
    
    def reset_profiler(self) -> None:
        """Reset the profiler."""
        self.profiler = cProfile.Profile()

class CacheManager:
    """Advanced caching system with TTL and disk persistence."""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live for cache entries in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.memory_cache = {}
        
    def _get_cache_path(self, key: str) -> Path:
        """Get filesystem path for a cache key."""
        return self.cache_dir / f"{hash(key)}.cache"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache."""
        # Check memory cache first
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.memory_cache[key]
            
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                mtime = cache_path.stat().st_mtime
                if time.time() - mtime < self.ttl:
                    with open(cache_path, 'rb') as f:
                        value = pd.read_pickle(f)
                        # Update memory cache
                        self.memory_cache[key] = (value, time.time())
                        return value
                else:
                    # Cache expired
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Error reading cache {key}: {e}")
                
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in cache."""
        # Update memory cache
        self.memory_cache[key] = (value, time.time())
        
        # Update disk cache
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pd.to_pickle(value, f)
        except Exception as e:
            logger.error(f"Error writing to cache {key}: {e}")
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        for file in self.cache_dir.glob("*.cache"):
            try:
                file.unlink()
            except Exception as e:
                logger.error(f"Error deleting cache file {file}: {e}")

def cache_result(ttl: int = 3600, maxsize: int = 128):
    """
    Decorator to cache function results with TTL.
    
    Args:
        ttl: Time-to-live in seconds
        maxsize: Maximum number of entries to keep in memory
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @lru_cache(maxsize=maxsize)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            # Check if cache is expired
            if hasattr(wrapper, '_last_run') and time.time() - wrapper._last_run < ttl:
                return result
            wrapper._last_run = time.time()
            return result
            
        wrapper.cache_clear = cached_func.cache_clear
        return wrapper
    return decorator

class ParallelProcessor:
    """Parallel processing utilities for CPU-bound tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes (default: CPU count)
        """
        self.max_workers = max_workers or (os.cpu_count() - 1 or 1)
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
    def parallel_apply(
        self,
        func: Callable[..., T],
        params_list: List[Dict[str, Any]],
        chunk_size: int = 100
    ) -> List[T]:
        """
        Apply a function to a list of parameter sets in parallel.
        
        Args:
            func: Function to parallelize
            params_list: List of parameter dictionaries
            chunk_size: Number of tasks per process
            
        Returns:
            List of results in the same order as params_list
        """
        results = [None] * len(params_list)
        
        # Process in chunks to avoid memory issues
        for i in range(0, len(params_list), chunk_size):
            chunk = params_list[i:i + chunk_size]
            futures = {
                self.executor.submit(func, **params): idx + i
                for idx, params in enumerate(chunk)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error in parallel task {idx}: {e}")
                    results[idx] = None
                    
        return results
    
    def parallel_apply_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.Series], T],
        n_jobs: int = -1,
        **kwargs
    ) -> pd.Series:
        """
        Apply a function to each row of a DataFrame in parallel.
        
        Args:
            df: Input DataFrame
            func: Function to apply to each row
            n_jobs: Number of jobs to run in parallel (-1 for all available CPUs)
            
        Returns:
            Series with the results
        """
        if n_jobs == -1:
            n_jobs = self.max_workers
            
        # Split the dataframe into chunks
        chunks = np.array_split(df, n_jobs)
        
        # Process each chunk in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self._process_chunk, chunk, func, **kwargs) 
                      for chunk in chunks]
            
            # Combine results
            results = []
            for future in as_completed(futures):
                results.append(future.result())
                
        return pd.concat(results)
    
    @staticmethod
    def _process_chunk(
        chunk: pd.DataFrame,
        func: Callable[[pd.Series], T],
        **kwargs
    ) -> pd.Series:
        """Process a chunk of the dataframe."""
        return chunk.apply(func, axis=1, **kwargs)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

# Example usage
if __name__ == "__main__":
    # Example of using the performance profiler
    profiler = PerformanceProfiler()
    
    @profiler.profile_function
    def example_function(n):
        return sum(i * i for i in range(n))
    
    # Profile the function
    result = example_function(1000000)
    print(profiler.get_profile_stats())
    
    # Example of using the cache manager
    cache = CacheManager()
    data = cache.get("expensive_operation")
    if data is None:
        print("Cache miss - performing expensive operation")
        data = np.random.rand(1000, 1000)  # Simulate expensive operation
        cache.set("expensive_operation", data)
    else:
        print("Cache hit - using cached data")
    
    # Example of parallel processing
    def process_row(row):
        return row.sum()  # Example row processing
    
    with ParallelProcessor() as pp:
        df = pd.DataFrame(np.random.rand(1000, 5))
        results = pp.parallel_apply_dataframe(df, process_row)
        print(f"Processed {len(results)} rows in parallel")