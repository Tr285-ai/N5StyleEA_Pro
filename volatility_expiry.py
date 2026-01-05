# volatility_expiry.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    HIGH_VOLATILITY = "High Volatility"
    MEDIUM_VOLATILITY = "Medium Volatility"
    LOW_VOLATILITY = "Low Volatility"
    EXTREME_VOLATILITY = "Extreme Volatility"

class VolatilityExpirySystem:
    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier_high: float = 2.0,
        atr_multiplier_medium: float = 1.0,
        min_expiry_seconds: int = 60,
        max_expiry_seconds: int = 300,  # 5 minutes
        default_expiry_seconds: int = 120  # 2 minutes
    ):
        self.atr_period = atr_period
        self.atr_multiplier_high = atr_multiplier_high
        self.atr_multiplier_medium = atr_multiplier_medium
        self.min_expiry_seconds = min_expiry_seconds
        self.max_expiry_seconds = max_expiry_seconds
        self.default_expiry_seconds = default_expiry_seconds
        self.volatility_history = []
        self.volatility_window = 20  # Number of periods to consider for volatility regime
        
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range (ATR) for volatility measurement."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def determine_market_regime(self, atr: float, atr_ma: float) -> MarketRegime:
        """Determine the current market volatility regime."""
        if pd.isna(atr) or pd.isna(atr_ma):
            return MarketRegime.MEDIUM_VOLATILITY
            
        atr_ratio = atr / atr_ma
        
        if atr_ratio > 2.0:
            return MarketRegime.EXTREME_VOLATILITY
        elif atr_ratio > 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif atr_ratio > 0.5:
            return MarketRegime.MEDIUM_VOLATILITY
        else:
            return MarketRegime.LOW_VOLATILITY
    
    def calculate_optimal_expiry(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        signal_direction: str
    ) -> Tuple[int, MarketRegime]:
        """
        Calculate optimal expiry time based on current market volatility.
        
        Returns:
            Tuple of (expiry_seconds, market_regime)
        """
        if len(market_data) < self.atr_period + 1:
            return self.default_expiry_seconds, MarketRegime.MEDIUM_VOLATILITY
            
        # Calculate ATR and its moving average
        atr = self.calculate_atr(market_data)
        current_atr = atr.iloc[-1]
        atr_ma = atr.rolling(window=self.atr_period).mean().iloc[-1]
        
        # Determine market regime
        regime = self.determine_market_regime(current_atr, atr_ma)
        
        # Base expiry time based on regime
        if regime == MarketRegime.EXTREME_VOLATILITY:
            base_expiry = self.min_expiry_seconds * 0.8  # Shorter expiry for extreme volatility
        elif regime == MarketRegime.HIGH_VOLATILITY:
            base_expiry = self.min_expiry_seconds * 1.0
        elif regime == MarketRegime.MEDIUM_VOLATILITY:
            base_expiry = (self.min_expiry_seconds + self.max_expiry_seconds) / 2
        else:  # LOW_VOLATILITY
            base_expiry = self.max_expiry_seconds * 0.9  # Longer expiry for low volatility
            
        # Adjust based on recent price action
        recent_volatility = self._analyze_recent_volatility(market_data)
        volatility_factor = self._calculate_volatility_factor(recent_volatility)
        
        # Calculate final expiry time
        expiry_seconds = int(base_expiry * volatility_factor)
        
        # Ensure expiry is within bounds
        expiry_seconds = max(self.min_expiry_seconds, min(expiry_seconds, self.max_expiry_seconds))
        
        return expiry_seconds, regime
    
    def _analyze_recent_volatility(self, data: pd.DataFrame, window: int = 5) -> float:
        """Analyze recent price action to detect sudden volatility changes."""
        if len(data) < window + 1:
            return 0.0
            
        recent = data.iloc[-window:]
        price_range = (recent['high'] - recent['low']).mean()
        avg_range = (data['high'] - data['low']).mean()
        
        if avg_range == 0:
            return 1.0
            
        return float(price_range / avg_range)
    
    def _calculate_volatility_factor(self, volatility_ratio: float) -> float:
        """Calculate the multiplier for expiry time based on volatility."""
        # Invert the relationship - higher volatility means shorter expiry
        return max(0.5, min(2.0, 1.0 / max(0.1, volatility_ratio)))