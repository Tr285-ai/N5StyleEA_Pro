# market_data.py
import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

logger = logging.getLogger("MarketData")

class MarketDataFeed:
    def __init__(self, broker_client, symbols: List[str], intervals: List[str] = ['1m', '5m', '15m', '1h', '4h', '1d']):
        """
        Initialize the market data feed.
        
        Args:
            broker_client: Instance of BrokerClient
            symbols: List of trading pairs to monitor
            intervals: List of time intervals for candles
        """
        self.broker = broker_client
        self.symbols = [s.upper() for s in symbols]
        self.intervals = intervals
        self.running = False
        self.data = {}
        self.callbacks = []
        self.lock = threading.Lock()
        self.thread = None
        
        # Initialize data structures
        self._init_data_structures()
        
    def _init_data_structures(self):
        """Initialize data storage"""
        for symbol in self.symbols:
            self.data[symbol] = {}
            for interval in self.intervals:
                self.data[symbol][interval] = {
                    'candles': pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']),
                    'last_update': None
                }
    
    def add_callback(self, callback: Callable):
        """Add a callback function to be called when new data arrives"""
        with self.lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback function"""
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)
    
    def _notify_callbacks(self, symbol: str, interval: str, candles: pd.DataFrame):
        """Notify all registered callbacks of new data"""
        with self.lock:
            for callback in self.callbacks:
                try:
                    callback(symbol, interval, candles)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__name__}: {e}")
    
    def start(self):
        """Start the market data feed"""
        if self.running:
            logger.warning("Market data feed is already running")
            return
            
        logger.info("Starting market data feed...")
        self.running = True
        
        # Initial data load
        self._load_initial_data()
        
        # Start background thread for updates
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        logger.info("Market data feed started")
    
    def stop(self):
        """Stop the market data feed"""
        logger.info("Stopping market data feed...")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("Market data feed stopped")
    
    def _load_initial_data(self):
        """Load initial historical data"""
        logger.info("Loading initial market data...")
        
        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    # Get historical klines
                    klines = self.broker.get_klines(symbol, interval, limit=1000)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert types
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Store in memory
                    with self.lock:
                        self.data[symbol][interval]['candles'] = df
                        self.data[symbol][interval]['last_update'] = datetime.utcnow()
                    
                    logger.debug(f"Loaded {len(df)} {interval} candles for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to load initial data for {symbol} {interval}: {e}")
    
    def _update_loop(self):
        """Background thread for updating market data"""
        logger.info("Starting market data update loop...")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Update each symbol and interval
                for symbol in self.symbols:
                    for interval in self.intervals:
                        try:
                            self._update_symbol_interval(symbol, interval)
                        except Exception as e:
                            logger.error(f"Error updating {symbol} {interval}: {e}")
                
                # Calculate sleep time to maintain ~1 second update interval
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0 - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in market data update loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _update_symbol_interval(self, symbol: str, interval: str):
        """Update data for a specific symbol and interval"""
        try:
            with self.lock:
                last_update = self.data[symbol][interval]['last_update']
                df = self.data[symbol][interval]['candles']
            
            # Determine how many new candles we need
            now = datetime.utcnow()
            if last_update is None:
                # If no data, get last 1000 candles
                limit = 1000
            else:
                # Otherwise, just get the latest candle
                limit = 1
            
            # Get latest klines
            klines = self.broker.get_klines(symbol, interval, limit=limit)
            
            if not klines:
                return
                
            # Convert to DataFrame
            new_data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            new_data[numeric_cols] = new_data[numeric_cols].apply(pd.to_numeric)
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
            
            with self.lock:
                if df.empty:
                    # First time loading data
                    df = new_data
                else:
                    # Append new data, removing duplicates
                    df = pd.concat([df, new_data]).drop_duplicates('timestamp', keep='last')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Update stored data
                self.data[symbol][interval]['candles'] = df
                self.data[symbol][interval]['last_update'] = now
                
                # Notify callbacks
                self._notify_callbacks(symbol, interval, df)
                
        except Exception as e:
            logger.error(f"Error updating {symbol} {interval}: {e}")
            raise
    
    def get_latest_candles(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Get the latest candles for a symbol and interval"""
        try:
            with self.lock:
                if symbol.upper() not in self.data or interval not in self.data[symbol.upper()]:
                    raise ValueError(f"No data available for {symbol} {interval}")
                    
                df = self.data[symbol.upper()][interval]['candles']
                return df.tail(limit).copy()
                
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol} {interval}: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str, interval: str, 
                               indicators: List[str] = ['sma_20', 'sma_50', 'rsi_14'], 
                               limit: int = 100) -> pd.DataFrame:
        """
        Calculate technical indicators for a symbol and interval
        
        Args:
            symbol: Trading pair
            interval: Time interval
            indicators: List of indicators to calculate
            limit: Number of candles to return
            
        Returns:
            DataFrame with OHLCV data and calculated indicators
        """
        try:
            # Get the candles
            df = self.get_latest_candles(symbol, interval, limit=limit)
            if df.empty:
                return df
                
            # Calculate indicators
            for indicator in indicators:
                try:
                    if indicator.startswith('sma_'):
                        period = int(indicator.split('_')[1])
                        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                        
                    elif indicator.startswith('ema_'):
                        period = int(indicator.split('_')[1])
                        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                        
                    elif indicator.startswith('rsi_'):
                        period = int(indicator.split('_')[1])
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                        rs = gain / loss
                        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                        
                    elif indicator.startswith('macd_'):
                        fast, slow, signal = map(int, indicator.split('_')[1:])
                        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
                        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
                        df['macd'] = exp1 - exp2
                        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
                        df['macd_hist'] = df['macd'] - df['macd_signal']
                        
                    elif indicator == 'bollinger':
                        df['bb_middle'] = df['close'].rolling(window=20).mean()
                        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
                        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
                        
                except Exception as e:
                    logger.error(f"Error calculating indicator {indicator}: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(self, symbol: str, interval: str = '1h', lookback: int = 24) -> dict:
        """
        Calculate market sentiment indicators
        
        Args:
            symbol: Trading pair
            interval: Time interval
            lookback: Number of periods to look back
            
        Returns:
            Dictionary with sentiment indicators
        """
        try:
            # Get OHLCV data
            df = self.get_latest_candles(symbol, interval, limit=lookback)
            if df.empty:
                return {}
                
            # Calculate simple sentiment indicators
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss).iloc[-1]))
            
            # Volume trend
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            
            # Price trend
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            price_above_sma = df['close'].iloc[-1] > sma_20 > sma_50
            
            # Sentiment score (0-100)
            sentiment_score = 50  # Neutral
            if rsi > 70:
                sentiment_score -= 20  # Overbought
            elif rsi < 30:
                sentiment_score += 20  # Oversold
                
            if volume_ratio > 1.5:
                sentiment_score += 10  # High volume confirms trend
                
            if price_above_sma:
                sentiment_score += 10  # Uptrend
            else:
                sentiment_score -= 10  # Downtrend
                
            # Clamp to 0-100
            sentiment_score = max(0, min(100, sentiment_score))
            
            return {
                'symbol': symbol,
                'interval': interval,
                'timestamp': datetime.utcnow().isoformat(),
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'price_above_sma': price_above_sma,
                'sentiment_score': sentiment_score,
                'sentiment': 'Bullish' if sentiment_score > 60 else 
                            'Bearish' if sentiment_score < 40 else 'Neutral'
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {
                'symbol': symbol,
                'interval': interval,
                'error': str(e)
            }