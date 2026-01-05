# signal_engine_v15_2.py
"""
Signal Engine v15.2 - Enhanced

A comprehensive signal generation system for financial markets that combines
multiple analysis techniques to generate trading signals with confidence scores.

Features:
- Multi-timeframe analysis
- Technical indicator integration
- Pattern recognition
- Volatility analysis
- Liquidity detection
- News sentiment integration
- Risk management
- Performance metrics
- Backtesting support

Author: N5StyleEA Team
Version: 15.2.0
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
import ta
import talib
from scipy import stats
import yfinance as yf  # For market data (can be replaced with your data source)
import requests
from requests.exceptions import RequestException
import pandas_ta as pta  # Additional technical indicators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('signal_engine.log')
    ]
)
logger = logging.getLogger('signal_engine')

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series
Array = np.ndarray

class SignalType(Enum):
    """Types of trading signals."""
    BUY = auto()
    SELL = auto()
    NEUTRAL = auto()
    STRONG_BUY = auto()
    STRONG_SELL = auto()
    CLOSE_LONG = auto()
    CLOSE_SHORT = auto()

class Timeframe(Enum):
    """Supported timeframes for analysis."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    @classmethod
    def to_pandas_freq(cls, tf: 'Timeframe') -> str:
        """Convert Timeframe to pandas frequency string."""
        return tf.value

@dataclass
class Signal:
    """A trading signal with metadata and confidence."""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    timeframe: str
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    source: str = "SignalEngine"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.name,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create Signal from dictionary."""
        return cls(
            symbol=data['symbol'],
            signal_type=SignalType[data['signal_type']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            timeframe=data['timeframe'],
            price=data['price'],
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            confidence=data.get('confidence', 0.5),
            source=data.get('source', 'SignalEngine'),
            metadata=data.get('metadata', {})
        )

class SignalEngine:
    """
    Advanced signal generation engine for financial markets.
    
    The SignalEngine analyzes market data across multiple timeframes and
    generates trading signals based on technical indicators, price action,
    volume analysis, and other quantitative factors.
    """
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        timeframes: Optional[List[Union[str, Timeframe]]] = None,
        config_path: Optional[Union[str, Path]] = None,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        max_drawdown: float = 0.2,     # 20% max drawdown
        enable_news: bool = True,
        enable_sentiment: bool = True,
        enable_ml: bool = True
    ):
        """
        Initialize the SignalEngine.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTC-USD')
            timeframes: List of timeframes to analyze
            config_path: Path to configuration file
            risk_free_rate: Annual risk-free rate for risk-adjusted returns
            max_drawdown: Maximum allowed drawdown (0-1)
            enable_news: Enable news-based signals
            enable_sentiment: Enable sentiment analysis
            enable_ml: Enable machine learning models
        """
        self.symbol = symbol.upper()
        self.timeframes = self._initialize_timeframes(timeframes)
        self.risk_free_rate = risk_free_rate
        self.max_drawdown = max_drawdown
        self.enable_news = enable_news
        self.enable_sentiment = enable_sentiment
        self.enable_ml = enable_ml
        
        # Initialize components
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.models = {}
        self.indicators = {}
        self.news_client = None
        self.sentiment_analyzer = None
        self.ml_model = None
        
        # Initialize technical indicators
        self._initialize_indicators()
        
        # Load ML models if enabled
        if self.enable_ml:
            self._load_ml_models()
            
        logger.info(f"Initialized SignalEngine for {self.symbol}")
    
    def _initialize_timeframes(
        self, 
        timeframes: Optional[List[Union[str, Timeframe]]]
    ) -> List[Timeframe]:
        """Initialize and validate timeframes."""
        if timeframes is None:
            return [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
            
        validated = []
        for tf in timeframes:
            if isinstance(tf, str):
                try:
                    validated.append(Timeframe[tf.upper()])
                except KeyError:
                    logger.warning(f"Invalid timeframe: {tf}. Skipping.")
            elif isinstance(tf, Timeframe):
                validated.append(tf)
                
        return validated or [Timeframe.M15, Timeframe.H1]
    
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'indicators': {
                'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'window': 20, 'window_dev': 2},
                'atr': {'window': 14},
                'stoch': {'k_window': 14, 'd_window': 3, 'smooth': 3},
                'volume': {'window': 20}
            },
            'signals': {
                'min_confidence': 0.6,
                'max_risk_reward': 3.0,
                'min_risk_reward': 1.5,
                'position_sizing': 'kelly',  # 'fixed', 'kelly', 'optimal_f'
                'max_position_size': 0.1,    # 10% of portfolio
                'volatility_adjust': True,
                'use_trailing_stop': True,
                'trailing_stop_pct': 0.02    # 2% trailing stop
            },
            'ml': {
                'enabled': True,
                'model_path': 'models/signal_ml_model.pkl',
                'features': ['rsi', 'macd', 'bb_width', 'atr', 'volume_ma']
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                
        return default_config
    
    def _initialize_indicators(self) -> None:
        """Initialize technical indicators."""
        self.indicators = {
            'rsi': lambda df: ta.momentum.RSIIndicator(
                close=df['close'],
                window=self.config['indicators']['rsi']['period']
            ).rsi(),
            'macd': lambda df: ta.trend.MACD(
                close=df['close'],
                window_fast=self.config['indicators']['macd']['fast'],
                window_slow=self.config['indicators']['macd']['slow'],
                window_sign=self.config['indicators']['macd']['signal']
            ),
            'bollinger': lambda df: ta.volatility.BollingerBands(
                close=df['close'],
                window=self.config['indicators']['bollinger']['window'],
                window_dev=self.config['indicators']['bollinger']['window_dev']
            ),
            'atr': lambda df: ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.config['indicators']['atr']['window']
            ).average_true_range(),
            'stoch': lambda df: ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.config['indicators']['stoch']['k_window'],
                smooth_window=self.config['indicators']['stoch']['smooth_window'],
                window_sign=self.config['indicators']['stoch']['d_window']
            ),
            'volume_ma': lambda df: ta.trend.SMAIndicator(
                close=df['volume'],
                window=self.config['indicators']['volume']['window']
            ).sma_indicator()
        }
    
    def _load_ml_models(self) -> None:
        """Load pre-trained machine learning models."""
        model_path = self.config['ml'].get('model_path')
        if model_path and os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading ML model: {e}")
                self.ml_model = None
        else:
            logger.warning(f"ML model not found at {model_path}")
    
    def get_historical_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Union[str, Timeframe] = Timeframe.H1,
        periods: int = 1000,
        end_date: Optional[Union[str, datetime]] = None,
        from_file: Optional[Union[str, Path]] = None
    ) -> DataFrame:
        """
        Get historical price data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            periods: Number of periods to retrieve
            end_date: End date for the data
            from_file: Load data from a file instead of API
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        symbol = symbol or self.symbol
        tf = Timeframe(timeframe) if isinstance(timeframe, str) else timeframe
        
        try:
            if from_file and os.path.exists(from_file):
                df = pd.read_csv(from_file, parse_dates=['date'], index_col='date')
                df = df.sort_index()
                logger.info(f"Loaded {len(df)} rows from {from_file}")
                return df[-periods:]  # Return most recent periods
            
            # Use yfinance as a fallback (replace with your data source)
            yf_symbol = symbol.replace('/', '-')
            df = yf.download(
                yf_symbol,
                period=f"{periods}{tf.value}",
                interval=tf.value,
                progress=False
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
                
            # Rename columns to lowercase
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Calculate indicators
            df = self._add_indicators(df)
            
            logger.info(f"Retrieved {len(df)} rows of {symbol} {tf.name} data")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _add_indicators(self, df: DataFrame) -> DataFrame:
        """Add technical indicators to the DataFrame."""
        if df.empty:
            return df
            
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Calculate all indicators
        for name, indicator in self.indicators.items():
            try:
                if name == 'macd':
                    macd = indicator(df)
                    df['macd'] = macd.macd()
                    df['macd_signal'] = macd.macd_signal()
                    df['macd_diff'] = macd.macd_diff()
                elif name == 'bollinger':
                    bb = indicator(df)
                    df['bb_high'] = bb.bollinger_hband()
                    df['bb_mid'] = bb.bollinger_mavg()
                    df['bb_low'] = bb.bollinger_lband()
                    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
                elif name == 'stoch':
                    stoch = indicator(df)
                    df['stoch_k'] = stoch.stoch()
                    df['stoch_d'] = stoch.stoch_signal()
                else:
                    df[name] = indicator(df)
            except Exception as e:
                logger.warning(f"Error calculating {name} indicator: {e}")
                
        # Additional indicators using pandas_ta
        try:
            # ADX
            df['adx'] = pta.adx(df['high'], df['low'], df['close'])['ADX_14']
            
            # Ichimoku Cloud
            ichimoku = pta.ichimoku(df['high'], df['low'], df['close'])
            df = df.join(ichimoku[['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26']])
            
            # Volume indicators
            df['obv'] = pta.obv(df['close'], df['volume'])
            df['cmf'] = pta.cmf(df['high'], df['low'], df['close'], df['volume'])
            
            # Volatility
            df['kc'] = pta.kc(df['high'], df['low'], df['close'])
            df['ui'] = pta.uo(df['high'], df['low'], df['close'])
            
        except Exception as e:
            logger.warning(f"Error calculating additional indicators: {e}")
            
        return df
    
    def generate_signals(
        self,
        df: Optional[DataFrame] = None,
        timeframe: Union[str, Timeframe] = Timeframe.H1,
        symbol: Optional[str] = None,
        use_ml: Optional[bool] = None
    ) -> List[Signal]:
        """
        Generate trading signals based on the provided data.
        
        Args:
            df: DataFrame with OHLCV data. If None, fetches data automatically.
            timeframe: Timeframe for the analysis
            symbol: Trading symbol
            use_ml: Whether to use machine learning for signal generation
            
        Returns:
            List of Signal objects
        """
        symbol = symbol or self.symbol
        tf = Timeframe(timeframe) if isinstance(timeframe, str) else timeframe
        use_ml = use_ml if use_ml is not None else self.enable_ml
        
        try:
            # Get data if not provided
            if df is None or df.empty:
                df = self.get_historical_data(symbol, tf)
                if df.empty:
                    logger.warning(f"No data available for {symbol} {tf.name}")
                    return []
            
            # Ensure we have enough data
            if len(df) < 50:  # Minimum 50 periods for meaningful indicators
                logger.warning(f"Not enough data points ({len(df)}) for {symbol} {tf.name}")
                return []
            
            # Add indicators if not present
            required_indicators = ['rsi', 'macd', 'bb_width', 'atr', 'volume_ma']
            if not all(ind in df.columns for ind in required_indicators):
                df = self._add_indicators(df)
            
            # Get the latest data point
            latest = df.iloc[-1]
            
            # Initialize signals list
            signals = []
            
            # 1. Generate trend signals
            trend_signals = self._generate_trend_signals(df, tf)
            signals.extend(trend_signals)
            
            # 2. Generate momentum signals
            momentum_signals = self._generate_momentum_signals(df, tf)
            signals.extend(momentum_signals)
            
            # 3. Generate volatility signals
            vol_signals = self._generate_volatility_signals(df, tf)
            signals.extend(vol_signals)
            
            # 4. Generate volume signals
            volume_signals = self._generate_volume_signals(df, tf)
            signals.extend(volume_signals)
            
            # 5. Generate pattern recognition signals
            pattern_signals = self._generate_pattern_signals(df, tf)
            signals.extend(pattern_signals)
            
            # 6. Apply machine learning if enabled
            if use_ml and self.ml_model is not None:
                ml_signals = self._generate_ml_signals(df, tf)
                signals.extend(ml_signals)
            
            # 7. Filter and rank signals
            filtered_signals = self._filter_signals(signals)
            ranked_signals = self._rank_signals(filtered_signals)
            
            # 8. Add risk management (stop loss, take profit)
            final_signals = self._add_risk_management(ranked_signals, df)
            
            logger.info(f"Generated {len(final_signals)} signals for {symbol} {tf.name}")
            return final_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
            return []
    
    def _generate_trend_signals(self, df: DataFrame, tf: Timeframe) -> List[Signal]:
        """Generate trend-following signals."""
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Moving average crossovers
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            # Golden cross (bullish)
            if (latest['sma_50'] > latest['sma_200'] and 
                prev['sma_50'] <= prev['sma_200']):
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.STRONG_BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.75,
                    source='MA_Crossover',
                    metadata={'type': 'golden_cross'}
                ))
            
            # Death cross (bearish)
            elif (latest['sma_50'] < latest['sma_200'] and 
                  prev['sma_50'] >= prev['sma_200']):
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.STRONG_SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.75,
                    source='MA_Crossover',
                    metadata={'type': 'death_cross'}
                ))
        
        # ADX trend strength
        if 'adx' in df.columns:
            adx = latest['adx']
            if adx > 25:  # Strong trend
                if latest['close'] > latest['sma_200']:
                    signals.append(Signal(
                        symbol=self.symbol,
                        signal_type=SignalType.BUY,
                        timestamp=df.index[-1],
                        timeframe=tf.name,
                        price=latest['close'],
                        confidence=min(0.7, (adx - 25) / 50),  # Scale confidence with ADX
                        source='ADX_Trend',
                        metadata={'adx': adx}
                    ))
                else:
                    signals.append(Signal(
                        symbol=self.symbol,
                        signal_type=SignalType.SELL,
                        timestamp=df.index[-1],
                        timeframe=tf.name,
                        price=latest['close'],
                        confidence=min(0.7, (adx - 25) / 50),
                        source='ADX_Trend',
                        metadata={'adx': adx}
                    ))
        
        return signals
    
    def _generate_momentum_signals(self, df: DataFrame, tf: Timeframe) -> List[Signal]:
        """Generate momentum-based signals."""
        signals = []
        latest = df.iloc[-1]
        
        # RSI signals
        if 'rsi' in df.columns:
            rsi = latest['rsi']
            rsi_config = self.config['indicators']['rsi']
            
            # Overbought condition
            if rsi > rsi_config['overbought']:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=min(0.8, (rsi - rsi_config['overbought']) / 20),
                    source='RSI',
                    metadata={'rsi': rsi}
                ))
            
            # Oversold condition
            elif rsi < rsi_config['oversold']:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=min(0.8, (rsi_config['oversold'] - rsi) / 20),
                    source='RSI',
                    metadata={'rsi': rsi}
                ))
        
        # MACD signals
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd = latest['macd']
            signal = latest['macd_signal']
            prev_macd = df['macd'].iloc[-2]
            prev_signal = df['macd_signal'].iloc[-2]
            
            # Bullish crossover
            if macd > signal and prev_macd <= prev_signal:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.65,
                    source='MACD',
                    metadata={'macd': macd, 'signal': signal}
                ))
            
            # Bearish crossover
            elif macd < signal and prev_macd >= prev_signal:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.65,
                    source='MACD',
                    metadata={'macd': macd, 'signal': signal}
                ))
        
        return signals
    
    def _generate_volatility_signals(self, df: DataFrame, tf: Timeframe) -> List[Signal]:
        """Generate signals based on volatility."""
        signals = []
        latest = df.iloc[-1]
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_high', 'bb_low', 'bb_mid']):
            close = latest['close']
            bb_high = latest['bb_high']
            bb_low = latest['bb_low']
            bb_mid = latest['bb_mid']
            
            # Price near lower band (oversold)
            if close <= bb_low:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=close,
                    confidence=0.7,
                    source='BollingerBands',
                    metadata={'bb_high': bb_high, 'bb_low': bb_low, 'bb_mid': bb_mid}
                ))
            
            # Price near upper band (overbought)
            elif close >= bb_high:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=close,
                    confidence=0.7,
                    source='BollingerBands',
                    metadata={'bb_high': bb_high, 'bb_low': bb_low, 'bb_mid': bb_mid}
                ))
        
        # ATR for volatility-based position sizing
        if 'atr' in df.columns:
            atr = latest['atr']
            # Can be used for position sizing or stop-loss calculation
            pass
            
        return signals
    
    def _generate_volume_signals(self, df: DataFrame, tf: Timeframe) -> List[Signal]:
        """Generate signals based on volume analysis."""
        signals = []
        latest = df.iloc[-1]
        
        # Volume spike
        if 'volume_ma' in df.columns and 'volume' in df.columns:
            volume_ma = latest['volume_ma']
            volume = latest['volume']
            
            # Volume spike (2x moving average)
            if volume > 2 * volume_ma:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY if latest['close'] > latest['open'] else SignalType.SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.6,
                    source='VolumeSpike',
                    metadata={'volume': volume, 'volume_ma': volume_ma}
                ))
        
        # OBV (On-Balance Volume) trend
        if 'obv' in df.columns:
            obv = latest['obv']
            obv_ma = df['obv'].rolling(window=20).mean().iloc[-1]
            
            if obv > obv_ma and df['close'].iloc[-1] > df['close'].iloc[-2]:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.65,
                    source='OBV',
                    metadata={'obv': obv, 'obv_ma': obv_ma}
                ))
            elif obv < obv_ma and df['close'].iloc[-1] < df['close'].iloc[-2]:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.65,
                    source='OBV',
                    metadata={'obv': obv, 'obv_ma': obv_ma}
                ))
        
        return signals
    
    def _generate_pattern_signals(self, df: DataFrame, tf: Timeframe) -> List[Signal]:
        """Generate signals based on candlestick patterns."""
        signals = []
        latest = df.iloc[-1]
        
        # Use TA-Lib to detect candlestick patterns
        try:
            # Bullish patterns
            hammer = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']).iloc[-1]
            engulfing = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[-1]
            morning_star = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1]
            
            # Bearish patterns
            shooting_star = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1]
            evening_star = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1]
            
            # Bullish signals
            if hammer > 0:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.7,
                    source='Pattern',
                    metadata={'pattern': 'hammer'}
                ))
                
            if engulfing > 0:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.75,
                    source='Pattern',
                    metadata={'pattern': 'bullish_engulfing'}
                ))
                
            if morning_star > 0:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.STRONG_BUY,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.8,
                    source='Pattern',
                    metadata={'pattern': 'morning_star'}
                ))
                
            # Bearish signals
            if shooting_star > 0:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.7,
                    source='Pattern',
                    metadata={'pattern': 'shooting_star'}
                ))
                
            if evening_star > 0:
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.STRONG_SELL,
                    timestamp=df.index[-1],
                    timeframe=tf.name,
                    price=latest['close'],
                    confidence=0.8,
                    source='Pattern',
                    metadata={'pattern': 'evening_star'}
                ))
                
        except Exception as e:
            logger.warning(f"Error detecting candlestick patterns: {e}")
            
        return signals
    
    def _generate_ml_signals(self, df: DataFrame, tf: Timeframe) -> List[Signal]:
        """Generate signals using machine learning models."""
        if self.ml_model is None:
            return []
            
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(df)
            if features.empty:
                return []
                
            # Make predictions
            predictions = self.ml_model.predict_proba(features)
            
            # Create signals based on predictions
            signals = []
            latest = df.iloc[-1]
            
            # Assuming binary classification (0=SELL, 1=BUY)
            for i, (prob_neg, prob_pos) in enumerate(predictions):
                if prob_pos > 0.6:  # Confidence threshold
                    signals.append(Signal(
                        symbol=self.symbol,
                        signal_type=SignalType.BUY if prob_pos > 0.5 else SignalType.SELL,
                        timestamp=df.index[i],
                        timeframe=tf.name,
                        price=df['close'].iloc[i],
                        confidence=abs(prob_pos - 0.5) * 2,  # Scale to [0, 1]
                        source='ML_Model',
                        metadata={'probability': prob_pos}
                    ))
                    
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return []
    
    def _prepare_ml_features(self, df: DataFrame) -> DataFrame:
        """Prepare features for machine learning model."""
        try:
            # Ensure we have the required indicators
            df = self._add_indicators(df)
            
            # Select features based on config
            feature_cols = self.config['ml'].get('features', [])
            if not feature_cols:
                feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Create feature DataFrame
            features = df[feature_cols].copy()
            
            # Handle missing values
            features = features.fillna(method='ffill').fillna(0)
            
            # Scale features
            if hasattr(self, 'scaler'):
                features = pd.DataFrame(
                    self.scaler.transform(features),
                    columns=features.columns,
                    index=features.index
                )
                
            return features
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return pd.DataFrame()
    
    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter out low-confidence and conflicting signals."""
        if not signals:
            return []
            
        # Group signals by type and timeframe
        signals_by_type = {}
        for signal in signals:
            key = (signal.signal_type, signal.timeframe)
            if key not in signals_by_type:
                signals_by_type[key] = []
            signals_by_type[key].append(signal)
        
        # Keep only the highest confidence signal of each type
        filtered = []
        for signal_group in signals_by_type.values():
            if signal_group:
                # Sort by confidence (descending) and take the first one
                best_signal = max(signal_group, key=lambda x: x.confidence)
                if best_signal.confidence >= self.config['signals']['min_confidence']:
                    filtered.append(best_signal)
        
        return filtered
    
    def _rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals by confidence and other factors."""
        if not signals:
            return []
            
        # Simple ranking by confidence for now
        return sorted(signals, key=lambda x: x.confidence, reverse=True)
    
    def _add_risk_management(self, signals: List[Signal], df: DataFrame) -> List[Signal]:
        """Add stop-loss and take-profit levels to signals."""
        if not signals:
            return []
            
        latest = df.iloc[-1]
        atr = latest.get('atr', 0) * 2  # Use 2x ATR for stop distance
        
        for signal in signals:
            price = signal.price
            sl_pct = 0.02  # Default 2% stop-loss
            tp_pct = 0.04  # Default 4% take-profit
            
            # Adjust based on volatility
            if atr > 0:
                atr_pct = atr / price
                sl_pct = max(0.01, min(0.05, atr_pct * 1.5))  # Between 1% and 5%
                tp_pct = sl_pct * 2  # 2:1 risk-reward ratio
            
            # Set stop-loss and take-profit
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                signal.stop_loss = price * (1 - sl_pct)
                signal.take_profit = price * (1 + tp_pct)
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                signal.stop_loss = price * (1 + sl_pct)
                signal.take_profit = price * (1 - tp_pct)
            
            # Add metadata
            signal.metadata.update({
                'stop_loss_pct': sl_pct,
                'take_profit_pct': tp_pct,
                'risk_reward_ratio': tp_pct / sl_pct
            })
        
        return signals
    
    def backtest(
        self,
        df: DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission per trade
        slippage: float = 0.0005,   # 0.05% slippage per trade
        verbose
            def backtest(
        self,
        df: DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission per trade
        slippage: float = 0.0005,   # 0.05% slippage per trade
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Backtest the signal generation strategy.
        
        Args:
            df: DataFrame with OHLCV data
            initial_balance: Starting capital
            commission: Commission per trade (percentage)
            slippage: Slippage per trade (percentage)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with backtest results
        """
        if df.empty:
            return {}
            
        # Initialize variables
        balance = initial_balance
        position = 0.0
        entry_price = 0.0
        trades = []
        equity = [initial_balance]
        
        # Generate signals for the entire dataset
        for i in range(1, len(df)):
            current_df = df.iloc[:i+1].copy()
            signals = self.generate_signals(df=current_df)
            
            if not signals:
                equity.append(balance + (position * df['close'].iloc[i]))
                continue
                
            signal = signals[0]  # Take the strongest signal
            
            # Calculate position size (fixed fraction of capital)
            position_size = min(0.1, balance * 0.1)  # Max 10% per trade
            current_price = df['close'].iloc[i]
            
            # Apply slippage
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                fill_price = current_price * (1 + slippage)
            else:
                fill_price = current_price * (1 - slippage)
                
            # Calculate commission
            trade_commission = position_size * commission
            
            # Execute trade
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and position <= 0:
                # Close short if any
                if position < 0:
                    pnl = position * (entry_price - fill_price) - trade_commission
                    balance += pnl
                    trades.append({
                        'entry_time': df.index[i-1],
                        'exit_time': df.index[i],
                        'type': 'short',
                        'entry': entry_price,
                        'exit': fill_price,
                        'pnl': pnl,
                        'balance': balance
                    })
                
                # Open long
                position = position_size / fill_price
                entry_price = fill_price
                balance -= position_size + trade_commission
                
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] and position >= 0:
                # Close long if any
                if position > 0:
                    pnl = position * (fill_price - entry_price) - trade_commission
                    balance += pnl
                    trades.append({
                        'entry_time': df.index[i-1],
                        'exit_time': df.index[i],
                        'type': 'long',
                        'entry': entry_price,
                        'exit': fill_price,
                        'pnl': pnl,
                        'balance': balance
                    })
                
                # Open short
                position = -position_size / fill_price
                entry_price = fill_price
                balance -= position_size + trade_commission
                
            # Update equity curve
            equity.append(balance + (position * current_price))
            
        # Calculate performance metrics
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
        max_drawdown = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min()
        total_return = (equity[-1] / initial_balance - 1) * 100
        
        # Create results dictionary
        results = {
            'initial_balance': initial_balance,
            'final_balance': equity[-1],
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': (len([t for t in trades if t['pnl'] > 0]) / len(trades)) if trades else 0,
            'avg_trade': np.mean([t['pnl'] for t in trades]) if trades else 0,
            'trades': trades,
            'equity': equity
        }
        
        if verbose:
            print(f"Backtest Results for {self.symbol}")
            print("=" * 50)
            print(f"Initial Balance: ${initial_balance:,.2f}")
            print(f"Final Balance: ${equity[-1]:,.2f}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: {max_drawdown*100:.2f}%")
            print(f"Number of Trades: {len(trades)}")
            print(f"Win Rate: {results['win_rate']*100:.1f}%")
            
        return results

    def save_signals(self, signals: List[Signal], filepath: Union[str, Path]) -> bool:
        """Save signals to a JSON file."""
        try:
            signals_data = [s.to_dict() for s in signals]
            with open(filepath, 'w') as f:
                json.dump(signals_data, f, indent=2, default=str)
            logger.info(f"Saved {len(signals)} signals to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
            return False

    @classmethod
    def load_signals(cls, filepath: Union[str, Path]) -> List[Signal]:
        """Load signals from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                signals_data = json.load(f)
            return [Signal.from_dict(s) for s in signals_data]
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            return []

    def optimize_parameters(
        self,
        df: DataFrame,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize signal generation parameters using grid search.
        
        Args:
            df: DataFrame with OHLCV data
            param_grid: Dictionary of parameters to optimize
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and results
        """
        from sklearn.model_selection import ParameterGrid
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        logger.info(f"Starting parameter optimization with {total_combinations} combinations")
        
        for i, params in enumerate(param_combinations, 1):
            try:
                # Update configuration
                for key, value in params.items():
                    keys = key.split('.')
                    if len(keys) == 1:
                        self.config[keys[0]] = value
                    else:
                        section = self.config
                        for k in keys[:-1]:
                            if k not in section:
                                section[k] = {}
                            section = section[k]
                        section[keys[-1]] = value
                
                # Run backtest
                result = self.backtest(df, verbose=False)
                
                if not result or metric not in result:
                    continue
                    
                score = result[metric]
                results.append({
                    'params': params,
                    'score': score,
                    'results': result
                })
                
                # Update best parameters
                if score > best_score:
                    best_score = score
                    best_params = params
                
                logger.info(f"Progress: {i}/{total_combinations} - {metric}: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error with parameters {params}: {e}")
                continue
                
        logger.info(f"Optimization complete. Best {metric}: {best_score:.4f}")
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

def example_usage():
    """Example usage of the SignalEngine."""
    # Initialize the signal engine
    engine = SignalEngine(
        symbol="BTC-USD",
        timeframes=[Timeframe.H1, Timeframe.H4],
        enable_ml=True
    )
    
    # Get historical data
    df = engine.get_historical_data(periods=1000)
    
    if df.empty:
        print("No data available")
        return
        
    # Generate signals
    signals = engine.generate_signals(df)
    
    if not signals:
        print("No signals generated")
        return
        
    # Print signals
    print("\nGenerated Signals:")
    for signal in signals[:5]:  # Show first 5 signals
        print(f"{signal.timestamp} - {signal.signal_type.name} at {signal.price:.2f} "
              f"(Confidence: {signal.confidence:.2f})")
    
    # Run backtest
    print("\nRunning backtest...")
    results = engine.backtest(df)
    
    # Save signals
    engine.save_signals(signals, "signals.json")

if __name__ == "__main__":
    example_usage()