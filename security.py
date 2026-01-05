# security.py
import os
from functools import wraps
from typing import List, Optional, Dict, Any
import jwt
import datetime
from enum import Enum
from flask import request, jsonify
from dataclasses import dataclass, field

class Permission(Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_COMMAND = "execute_command"
    DEPLOY = "deploy"
    TRADE = "execute_trade"
    VIEW_SENSITIVE = "view_sensitive"

class SecurityManager:
    def __init__(self, secret_key: str, admin_users: Optional[List[str]] = None):
        if not secret_key or not isinstance(secret_key, str):
            raise ValueError("A valid secret key is required")
        self.secret_key = secret_key
        self.admin_users = set(admin_users or [])
        self.user_permissions: Dict[str, List[Permission]] = {}
        self.token_blacklist = set()
        
    def generate_token(self, user_id: str, permissions: Optional[List[Permission]] = None, 
                      expires_in: int = 3600) -> str:
        """Generate a JWT token for a user with specified permissions."""
        if not user_id or not isinstance(user_id, str):
            raise ValueError("User ID must be a non-empty string")
            
        payload = {
            'user_id': user_id,
            'permissions': [p.value for p in (permissions or [])],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in),
            'iat': datetime.datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
        
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify a JWT token and return the payload if valid."""
        if not token:
            raise ValueError("Token is required")
        if token in self.token_blacklist:
            raise PermissionError("Token has been revoked")
            
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise PermissionError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise PermissionError(f"Invalid token: {str(e)}")
            
    def revoke_token(self, token: str) -> None:
        """Add a token to the blacklist."""
        self.token_blacklist.add(token)
            
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        if user_id in self.admin_users:
            return True
        return permission.value in [p.value for p in self.user_permissions.get(user_id, [])]
        
    def require_permission(self, permission: Permission):
        """Decorator to require a specific permission for a function."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                token = request.headers.get('Authorization')
                if not token or not token.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid token'}), 401
                    
                try:
                    token = token.split(' ')[1]
                    payload = self.verify_token(token)
                    if not self.has_permission(payload['user_id'], permission):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                    return f(*args, **kwargs)
                except Exception as e:
                    return jsonify({'error': str(e)}), 401
            return decorated_function
        return decorator

    def token_required(self, f):
        """Decorator to require a valid token for a function."""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token or not token.startswith('Bearer '):
                return jsonify({'error': 'Token is missing'}), 401
            try:
                token = token.split(' ')[1]
                data = self.verify_token(token)
                request.user = data
            except Exception as e:
                return jsonify({'error': str(e)}), 401
            return f(*args, **kwargs)
        return decorated