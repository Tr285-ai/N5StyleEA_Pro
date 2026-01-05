import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

class TradingLogger:
    """
    Advanced logging utility for the trading system.
    Supports both console and file logging with rotation.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TradingLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(
        self,
        name: str = "trading",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        if self._initialized:
            return
            
        self.name = name
        self.log_level = log_level
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.logger = None
        
        self._setup_logger()
        self._initialized = True
        
    def _setup_logger(self) -> None:
        """Configure the logger with handlers and formatters."""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation if log file is specified
        if self.log_file:
            try:
                # Create log directory if it doesn't exist
                log_path = Path(self.log_file).parent
                log_path.mkdir(parents=True, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to set up file logging: {e}", file=sys.stderr)
                
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        if self.logger is None:
            self._setup_logger()
        return self.logger
        
    @classmethod
    def get_default_logger(cls) -> logging.Logger:
        """Get a default configured logger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.get_logger()

# Global logger instance
logger = TradingLogger().get_logger()

def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        log_level: Logging level (default: logging.INFO)
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    return TradingLogger(
        name=name,
        log_level=log_level,
        log_file=log_file
    ).get_logger()