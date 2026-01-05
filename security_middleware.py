from functools import wraps
from flask import request, jsonify
import time
import re
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, Type, Union, List
from functools import lru_cache
import os

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

class RateLimiter:
    """Thread-safe rate limiter implementation with automatic cleanup."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.access_records: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Clean up every 5 minutes
    
    def is_rate_limited(self, key: str) -> bool:
        """Check if the request should be rate limited.
        
        Args:
            key: The key to track (typically client IP or user ID)
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        current_time = time.time()
        
        with self._lock:
            # Periodic cleanup of old entries
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_entries()
                self._last_cleanup = current_time
            
            # Initialize if key doesn't exist
            if key not in self.access_records:
                self.access_records[key] = []
            
            # Remove old entries for this key
            self.access_records[key] = [
                t for t in self.access_records[key] 
                if current_time - t < self.window_seconds
            ]
            
            # Check if rate limit exceeded
            if len(self.access_records[key]) >= self.max_requests:
                return True
                
            # Add current request
            self.access_records[key].append(current_time)
            return False
    
    def _cleanup_old_entries(self) -> None:
        """Remove old entries from access records to prevent memory leaks."""
        current_time = time.time()
        keys_to_delete = []
        
        for key, timestamps in self.access_records.items():
            # Remove old timestamps
            valid_timestamps = [t for t in timestamps 
                              if current_time - t < self.window_seconds]
            
            if valid_timestamps:
                self.access_records[key] = valid_timestamps
            else:
                keys_to_delete.append(key)
        
        # Remove keys with no valid timestamps
        for key in keys_to_delete:
            self.access_records.pop(key, None)

# Initialize rate limiter with default values (100 requests per minute per IP)
def get_rate_limit_config() -> tuple[int, int]:
    """Get rate limit configuration from environment variables or use defaults."""
    try:
        max_requests = int(os.getenv('RATE_LIMIT_MAX_REQUESTS', '100'))
        window_seconds = int(os.getenv('RATE_LIMIT_WINDOW_SECONDS', '60'))
        return max_requests, window_seconds
    except (ValueError, TypeError):
        return 100, 60  # Fallback to defaults

# Initialize the rate limiter
rate_limiter = RateLimiter(*get_rate_limit_config())

def rate_limit(f: F) -> F:
    """Decorator to rate limit requests.
    
    Args:
        f: The view function to decorate
        
    Returns:
        The decorated function
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP, considering X-Forwarded-For if behind a proxy
        if 'X-Forwarded-For' in request.headers:
            client_ip = request.headers['X-Forwarded-For'].split(',')[0].strip()
        else:
            client_ip = request.remote_addr or 'unknown'
            
        if rate_limiter.is_rate_limited(client_ip):
            return jsonify({
                'status': 'error',
                'code': 'rate_limit_exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429
            
        # Add rate limit headers to the response
        response = f(*args, **kwargs)
        if not isinstance(response, tuple):
            response = (response, 200)
            
        # Add rate limit headers (RFC 6585)
        response_headers = {
            'X-RateLimit-Limit': str(rate_limiter.max_requests),
            'X-RateLimit-Remaining': str(rate_limiter.max_requests - 1),  # Approximate
            'X-RateLimit-Reset': str(int(time.time() + rate_limiter.window_seconds))
        }
        
        if len(response) == 2:
            resp, status = response
            if hasattr(resp, 'headers'):
                for k, v in response_headers.items():
                    resp.headers[k] = v
            return resp, status
        return response
        
    return decorated_function  # type: ignore

def validate_input(
    input_data: Dict[str, Any], 
    rules: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, str]]:
    """Validate input data against a set of rules.
    
    Args:
        input_data: Dictionary of input data to validate
        rules: Validation rules for each field
        
    Returns:
        Dict of validation errors or None if validation passes
        
    Example rules:
    {
        'username': {
            'type': str,
            'required': True,
            'min_length': 3,
            'max_length': 50,
            'regex': r'^[a-zA-Z0-9_]+$',
            'custom': lambda x: x != 'admin' or 'Cannot use reserved username'
        },
        'email': {
            'type': str,
            'required': True,
            'regex': r'^[^@]+@[^@]+\.[^@]+$',
            'normalize': lambda x: x.lower().strip()
        },
        'age': {
            'type': int,
            'required': False,
            'min': 18,
            'max': 120
        }
    }
    """
    errors: Dict[str, str] = {}
    
    for field, rule in rules.items():
        value = input_data.get(field)
        
        # Check if field is required
        if rule.get('required', False) and (value is None or value == ''):
            errors[field] = rule.get('required_message', 'This field is required')
            continue
            
        # Skip further checks if value is None/empty and not required
        if value is None or value == '':
            input_data[field] = None
            continue
            
        # Type conversion and validation
        if 'type' in rule:
            try:
                if rule['type'] == bool and isinstance(value, str):
                    # Special handling for boolean strings
                    value = value.lower() in ('true', '1', 'yes', 'y', 't')
                elif rule['type'] != type(None):  # Skip None type conversion
                    value = rule['type'](value)
                input_data[field] = value  # Update with converted value
            except (ValueError, TypeError, AttributeError) as e:
                errors[field] = rule.get('type_message', 
                    f'Must be of type {rule["type"].__name__}')
                continue
                
        # Apply normalization if specified
        if 'normalize' in rule and value is not None:
            try:
                normalized = rule['normalize'](value)
                input_data[field] = normalized
                value = normalized
            except Exception as e:
                errors[field] = 'Invalid value format'
                continue
        
        # String validations
        if isinstance(value, str):
            value = value.strip()
            input_data[field] = value
            
            if 'min_length' in rule and len(value) < rule['min_length']:
                errors[field] = rule.get('min_length_message',
                    f'Must be at least {rule["min_length"]} characters')
                
            if 'max_length' in rule and len(value) > rule['max_length']:
                errors[field] = rule.get('max_length_message',
                    f'Must be at most {rule["max_length"]} characters')
                
            if 'regex' in rule and not re.match(rule['regex'], value):
                errors[field] = rule.get('regex_message', 'Invalid format')
                
            if 'choices' in rule and value not in rule['choices']:
                errors[field] = rule.get('choices_message',
                    f'Must be one of: {", ".join(map(str, rule["choices"]))}')
        
        # Numeric validations
        if isinstance(value, (int, float)):
            if 'min' in rule and value < rule['min']:
                errors[field] = rule.get('min_message',
                    f'Must be at least {rule["min"]}')
                
            if 'max' in rule and value > rule['max']:
                errors[field] = rule.get('max_message',
                    f'Must be at most {rule["max"]}')
        
        # Custom validation function
        if 'custom' in rule and value is not None:
            try:
                result = rule['custom'](value)
                if isinstance(result, str):
                    errors[field] = result
                elif result is not None and not result:
                    errors[field] = rule.get('custom_message', 'Invalid value')
            except Exception as e:
                errors[field] = 'Validation failed for this field'
    
    return errors if errors else None

def input_validation(
    rules: Dict[str, Dict[str, Any]],
    locations: Optional[List[str]] = None,
    allow_unknown: bool = False,
    strip_whitespace: bool = True
) -> Callable[[F], F]:
    """Decorator to validate request data against validation rules.
    
    Args:
        rules: Validation rules for each field
        locations: Where to look for data ('json', 'form', 'args', 'headers', 'cookies')
        allow_unknown: If True, allows fields not in rules
        strip_whitespace: If True, strip whitespace from string values
        
    Returns:
        Decorated view function
    """
    if locations is None:
        locations = ['json', 'form', 'args']
        
    def decorator(f: F) -> F:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get data from specified locations
            data = {}
            
            if 'json' in locations and request.is_json:
                json_data = request.get_json(silent=True) or {}
                if isinstance(json_data, dict):
                    data.update(json_data)
            
            if 'form' in locations and request.form:
                data.update(request.form.to_dict())
                
            if 'args' in locations and request.args:
                data.update(request.args.to_dict())
                
            if 'headers' in locations:
                for header, value in request.headers.items():
                    # Convert header names to lowercase with underscores
                    header_key = header.lower().replace('-', '_')
                    data[f'header_{header_key}'] = value
                    
            if 'cookies' in locations:
                data.update(request.cookies)
            
            # Filter data to only include fields in rules if not allowing unknown
            if not allow_unknown:
                data = {k: v for k, v in data.items() if k in rules}
            
            # Strip whitespace from string values if enabled
            if strip_whitespace:
                for key, value in data.items():
                    if isinstance(value, str):
                        data[key] = value.strip()
            
            # Validate data
            errors = validate_input(data, rules)
            if errors:
                return jsonify({
                    'status': 'error',
                    'code': 'validation_error',
                    'message': 'One or more validation errors occurred',
                    'errors': errors
                }), 400
                
            # Add validated data to kwargs
            kwargs['validated_data'] = data
            return f(*args, **kwargs)
            
        return decorated_function  # type: ignore
    return decorator