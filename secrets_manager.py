# security/secrets_manager.py
import os
import json
from typing import Dict, Optional
import logging
from pathlib import Path
import keyring
from cryptography.fernet import Fernet
import base64
import hashlib

logger = logging.getLogger(__name__)

class SecretsManager:
    """Secure management of application secrets with local encryption."""
    
    def __init__(self, app_name: str = "n5style_ea", config_dir: Optional[Path] = None):
        """
        Initialize the secrets manager.
        
        Args:
            app_name: Application name for namespacing
            config_dir: Directory to store encrypted secrets (default: ~/.config/{app_name})
        """
        self.app_name = app_name
        self.config_dir = config_dir or Path.home() / ".config" / app_name
        self.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Initialize encryption
        self.key_file = self.config_dir / ".encryption_key"
        self._initialize_encryption_key()
        
    def _initialize_encryption_key(self):
        """Initialize or load the encryption key."""
        if not self.key_file.exists():
            # Generate new key
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)  # Restrict permissions
        else:
            key = self.key_file.read_bytes()
            
        self.cipher_suite = Fernet(key)
        
    def _get_service_name(self, service: str) -> str:
        """Generate a service name for keyring."""
        return f"{self.app_name}_{service}"
        
    def _get_keyring_username(self, key: str) -> str:
        """Generate a deterministic username for keyring entries."""
        return hashlib.sha256(f"{self.app_name}_{key}".encode()).hexdigest()
        
    def set_secret(self, key: str, value: str, use_keyring: bool = True) -> None:
        """
        Store a secret securely.
        
        Args:
            key: Secret identifier
            value: Secret value
            use_keyring: Whether to use system keyring (more secure) or encrypted file
        """
        if use_keyring:
            try:
                service = self._get_service_name("secrets")
                username = self._get_keyring_username(key)
                keyring.set_password(service, username, value)
                logger.debug(f"Stored secret '{key}' in system keyring")
            except Exception as e:
                logger.warning(f"Failed to store in keyring, falling back to file storage: {e}")
                use_keyring = False
                
        if not use_keyring:
            secrets = self._load_secrets_file()
            secrets[key] = value
            self._save_secrets_file(secrets)
            logger.debug(f"Stored secret '{key}' in encrypted file")
            
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a secret.
        
        Args:
            key: Secret identifier
            default: Default value if secret not found
            
        Returns:
            The secret value or default if not found
        """
        # Try keyring first
        try:
            service = self._get_service_name("secrets")
            username = self._get_keyring_username(key)
            value = keyring.get_password(service, username)
            if value is not None:
                return value
        except Exception as e:
            logger.debug(f"Keyring access failed: {e}")
            
        # Fall back to file storage
        secrets = self._load_secrets_file()
        return secrets.get(key, default)
        
    def _load_secrets_file(self) -> Dict[str, str]:
        """Load and decrypt the secrets file."""
        secrets_file = self.config_dir / "secrets.enc"
        if not secrets_file.exists():
            return {}
            
        try:
            encrypted_data = secrets_file.read_bytes()
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt secrets file: {e}")
            return {}
            
    def _save_secrets_file(self, secrets: Dict[str, str]) -> None:
        """Encrypt and save the secrets file."""
        secrets_file = self.config_dir / "secrets.enc"
        try:
            data = json.dumps(secrets).encode()
            encrypted_data = self.cipher_suite.encrypt(data)
            secrets_file.write_bytes(encrypted_data)
            secrets_file.chmod(0o600)  # Restrict permissions
        except Exception as e:
            logger.error(f"Failed to save secrets file: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    # Initialize with application name
    secrets = SecretsManager()
    
    # Store a secret
    secrets.set_secret("BINANCE_API_KEY", "your-api-key-here")
    
    # Retrieve a secret
    api_key = secrets.get_secret("BINANCE_API_KEY")
    print(f"Retrieved API key: {api_key}")