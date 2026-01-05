from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.logger = logging.getLogger(f"strategy.{name}")
        
    async def initialize(self) -> None:
        """Initialize the strategy with required resources."""
        if self.is_initialized:
            return
            
        try:
            await self._initialize_resources()
            self.is_initialized = True
            self.logger.info(f"Strategy '{self.name}' initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize strategy: {e}")
            raise

    @abstractmethod
    async def _initialize_resources(self) -> None:
        """Initialize any resources needed by the strategy."""
        pass
        
    @abstractmethod
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data: Market data in pandas DataFrame format
            
        Returns:
            Dictionary containing analysis results and signals
        """
        pass
        
    @abstractmethod
    async def execute(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trades based on signals.
        
        Args:
            signals: Dictionary containing trading signals
            
        Returns:
            Dictionary with execution results
        """
        pass
        
    async def shutdown(self) -> None:
        """Clean up resources used by the strategy."""
        self.is_initialized = False
        self.logger.info(f"Strategy '{self.name}' shut down")
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

class BaseIndicator(ABC):
    """Base class for technical indicators."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator values.
        
        Args:
            data: Input data as a pandas DataFrame
            
        Returns:
            pandas Series with the calculated values
        """
        pass