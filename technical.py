"""
Technical Analysis Module

Provides technical indicators and analysis tools for trading strategies.
Includes various technical indicators commonly used in algorithmic trading.

Author: N5StyleEA Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import talib
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class IndicatorType(Enum):
    """Types of technical indicators"""
    TREND = auto()
    MOMENTUM = auto()
    VOLATILITY = auto()
    VOLUME = auto()

@dataclass
class IndicatorResult:
    """Container for indicator results"""
    values: np.ndarray
    signals: np.ndarray
    metadata: Dict[str, any]
    indicator_type: IndicatorType

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    def __init__(self, ohlcv_data: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = ohlcv_data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate input data structure"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")
    
    def calculate_rsi(self, period: int = 14) -> IndicatorResult:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            period: RSI period (default: 14)
            
        Returns:
            IndicatorResult with RSI values and signals
        """
        try:
            rsi = talib.RSI(self.data['close'], timeperiod=period)
            signals = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
            
            return IndicatorResult(
                values=rsi,
                signals=signals,
                metadata={'period': period, 'overbought': 70, 'oversold': 30},
                indicator_type=IndicatorType.MOMENTUM
            )
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            raise
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, np.ndarray]:
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            
        Returns:
            Dict containing MACD line, signal line, and histogram
        """
        try:
            macd, signal, hist = talib.MACD(
                self.data['close'],
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            
            return {
                'macd': macd,
                'signal': signal,
                'histogram': hist
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            raise
    
    def calculate_bollinger_bands(self, period: int = 20, num_std: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Calculate Bollinger Bands
        
        Args:
            period: MA period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            
        Returns:
            Dict containing upper, middle, and lower bands
        """
        try:
            upper, middle, lower = talib.BBANDS(
                self.data['close'],
                timeperiod=period,
                nbdevup=num_std,
                nbdevdn=num_std
            )
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            raise
    
    def calculate_atr(self, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR)
        
        Args:
            period: ATR period (default: 14)
            
        Returns:
            ATR values
        """
        try:
            return talib.ATR(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                timeperiod=period
            )
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            raise
    
    def calculate_ichimoku(self) -> Dict[str, np.ndarray]:
        """
        Calculate Ichimoku Cloud components
        
        Returns:
            Dict containing all Ichimoku components
        """
        try:
            # Tenkan-sen (Conversion Line)
            high_9 = self.data['high'].rolling(window=9).max()
            low_9 = self.data['low'].rolling(window=9).min()
            tenkan_sen = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            high_26 = self.data['high'].rolling(window=26).max()
            low_26 = self.data['low'].rolling(window=26).min()
            kijun_sen = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            high_52 = self.data['high'].rolling(window=52).max()
            low_52 = self.data['low'].rolling(window=52).min()
            senkou_span_b = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            chikou_span = self.data['close'].shift(-26)
            
            return {
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span
            }
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Example data
    data = {
        'open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
        'high': [101, 102, 103, 104, 105, 106, 105, 104, 103, 102],
        'low': [99, 100, 101, 102, 103, 104, 103, 102, 101, 100],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 104.5, 103.5, 102.5, 101.5],
        'volume': [1000, 1200, 1300, 1100, 1400, 1500, 1300, 1200, 1100, 1000]
    }
    
    df = pd.DataFrame(data)
    ta = TechnicalIndicators(df)
    
    # Calculate RSI
    rsi = ta.calculate_rsi(14)
    print("RSI:", rsi.values)
    
    # Calculate MACD
    macd = ta.calculate_macd()
    print("MACD:", macd)
    
    # Calculate Bollinger Bands
    bb = ta.calculate_bollinger_bands()
    print("Bollinger Bands:", bb)