# optimizations/jit_manager.py
from numba import jit, njit, types
from numba.extending import register_jitable
from functools import wraps
import time
import logging
from typing import Callable, Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class JITManager:
    """Manages JIT compilation and caching of functions."""
    
    def __init__(self, cache_enabled: bool = True, debug: bool = False):
        """
        Initialize the JIT manager.
        
        Args:
            cache_enabled: Whether to cache compiled functions
            debug: Enable debug mode for JIT compilation
        """
        self.cache_enabled = cache_enabled
        self.debug = debug
        self.compilation_cache = {}
        self.compilation_times = {}
        
    def jit(self, func=None, **kwargs):
        """
        Decorator for JIT compilation with caching.
        
        Args:
            **kwargs: Additional arguments to pass to numba.jit
            
        Returns:
            Decorated function
        """
        if func is None:
            return lambda f: self.jit(f, **kwargs)
            
        cache_key = self._get_cache_key(func, kwargs)
        
        if self.cache_enabled and cache_key in self.compilation_cache:
            return self.compilation_cache[cache_key]
            
        # Add debug options if enabled
        if self.debug:
            kwargs.setdefault('debug', True)
            kwargs.setdefault('boundscheck', True)
            
        # Compile the function
        start_time = time.time()
        jitted_func = jit(func, **kwargs)
        compile_time = time.time() - start_time
        
        self.compilation_times[cache_key] = compile_time
        logger.debug(f"Compiled {func.__name__} in {compile_time:.4f}s")
        
        if self.cache_enabled:
            self.compilation_cache[cache_key] = jitted_func
            
        return jitted_func
    
    def njit(self, func=None, **kwargs):
        """Convenience method for @njit with caching."""
        if func is None:
            return lambda f: self.njit(f, **kwargs)
        return self.jit(func, nopython=True, **kwargs)
    
    def _get_cache_key(self, func: Callable, options: Dict[str, Any]) -> str:
        """Generate a cache key for a function and its compilation options."""
        return f"{func.__name__}_{hash(frozenset(options.items()))}"
    
    def get_compilation_stats(self) -> Dict[str, float]:
        """Get statistics about JIT compilation times."""
        return self.compilation_times
    
    def clear_cache(self) -> None:
        """Clear the JIT compilation cache."""
        self.compilation_cache.clear()
        self.compilation_times.clear()

# Global instance
jit_manager = JITManager()

# Convenience decorators
def optimized_jit(func=None, **kwargs):
    """Decorator for optimized JIT compilation with caching."""
    return jit_manager.jit(func, **kwargs)

def optimized_njit(func=None, **kwargs):
    """Decorator for optimized nopython JIT compilation with caching."""
    return jit_manager.njit(func, **kwargs)