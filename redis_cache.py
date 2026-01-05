# caching/redis_cache.py
import json
import logging
from typing import Any, Optional, Dict, Union, TypeVar, Callable
import redis
from functools import wraps
import pickle
import hashlib
import asyncio
from datetime import timedelta

logger = logging.getLogger(__name__)
T = TypeVar('T')

class RedisCache:
    """Redis-based caching system with TTL and serialization support."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/1",
        default_ttl: int = 3600,
        prefix: str = "cache:"
    ):
        """
        Initialize the Redis cache.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            prefix: Prefix for all cache keys
        """
        self.redis = redis.Redis.from_url(redis_url)
        self.default_ttl = default_ttl
        self.prefix = prefix
        
    def _get_key(self, key: str) -> str:
        """Get the full cache key with prefix."""
        return f"{self.prefix}{key}"
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize the value
            
        Returns:
            bool: True if successful
        """
        try:
            cache_key = self._get_key(key)
            ttl = ttl if ttl is not None else self.default_ttl
            
            if serialize:
                value = pickle.dumps(value)
                
            return bool(
                self.redis.set(cache_key, value, ex=ttl)
            )
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def get(
        self,
        key: str,
        default: Any = None,
        deserialize: bool = True
    ) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key doesn't exist
            deserialize: Whether to deserialize the value
            
        Returns:
            Cached value or default
        """
        try:
            cache_key = self._get_key(key)
            value = self.redis.get(cache_key)
            
            if value is None:
                return default
                
            if deserialize:
                try:
                    value = pickle.loads(value)
                except (pickle.PickleError, AttributeError):
                    pass
                    
            return value
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return default
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys from the cache."""
        if not keys:
            return 0
        return self.redis.delete(*[self._get_key(k) for k in keys])
    
    async def clear(self, pattern: str = "*") -> int:
        """Clear all keys matching the pattern."""
        keys = self.redis.keys(self._get_key(pattern))
        if not keys:
            return 0
        return await self.delete(*[k.decode() for k in keys])
    
    async def get_or_set(
        self,
        key: str,
        default: Union[Any, Callable[[], T]],
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> T:
        """
        Get a value from cache, or set it if it doesn't exist.
        
        Args:
            key: Cache key
            default: Default value or callable that returns the default value
            ttl: Time to live in seconds
            serialize: Whether to serialize the value
            
        Returns:
            Cached or default value
        """
        value = await self.get(key, deserialize=serialize)
        if value is not None:
            return value
            
        if callable(default):
            value = default()
        else:
            value = default
            
        await self.set(key, value, ttl=ttl, serialize=serialize)
        return value
    
    def cached(
        self,
        ttl: int = 300,
        key_prefix: str = "",
        key_func: Optional[Callable[..., str]] = None
    ) -> Callable:
        """
        Decorator to cache function results.
        
        Args:
            ttl: Time to live in seconds
            key_prefix: Prefix for cache keys
            key_func: Function to generate cache keys
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Generate a key based on function name and arguments
                    key_parts = [key_prefix, func.__name__]
                    key_parts.extend([str(arg) for arg in args])
                    key_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Call the function
                result = await func(*args, **kwargs)
                
                # Cache the result
                await self.set(cache_key, result, ttl=ttl)
                
                return result
            return wrapper
        return decorator