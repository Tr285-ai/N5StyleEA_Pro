import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class FeatureEngineer:
    """
    Feature engineering class for financial time series data.
    Handles technical indicators, time features, and data transformations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FeatureEngineer with optional configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.feature_cache = {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must contain columns: {required}")
            
        df = df.copy()
        
        try:
            # Ensure numeric columns
            for col in required + (['volume'] if 'volume' in df.columns else []):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Handle NaN values
            df = self._handle_missing_values(df)
            
            # Add technical indicators
            df = self._add_trend_indicators(df)
            df = self._add_momentum_indicators(df)
            df = self._add_volatility_indicators(df)
            
            if 'volume' in df.columns:
                df = self._add_volume_indicators(df)
                
        except Exception as e:
            raise RuntimeError(f"Error adding technical indicators: {str(e)}")
            
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # Forward fill then backfill any remaining NaNs
        return df.ffill().bfill()
        
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        return df
        
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        return df
        
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        df['atr'] = talib.ATR(
            df['high'], df['low'], df['close'], timeperiod=14
        )
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        return df
        
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['adl'] = talib.AD(
            df['high'], df['low'], df['close'], df['volume']
        )
        return df
        
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the dataframe.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
            
        df = df.copy()
        
        # Time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7.0)
        
        return df
        
    @staticmethod
    def create_sequences(
        data: np.ndarray, 
        targets: Optional[np.ndarray] = None,
        seq_length: int = 60, 
        step: int = 1
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Create sequences of data for time series prediction.
        
        Args:
            data: Input features (n_samples, n_features)
            targets: Target values (n_samples,)
            seq_length: Length of each sequence
            step: Step size between sequences
            
        Returns:
            Tuple of (sequences, targets) or just sequences if targets is None
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
            
        if len(data.shape) != 2:
            raise ValueError("Data must be 2D (samples, features)")
            
        n_samples = data.shape[0]
        n_features = data.shape[1]
        sequences = []
        target_sequences = []
        
        for i in range(0, n_samples - seq_length, step):
            sequences.append(data[i:i + seq_length])
            if targets is not None:
                target_sequences.append(targets[i + seq_length - 1])
                
        sequences = np.array(sequences)
        
        if targets is not None:
            return sequences, np.array(target_sequences)
        return sequences