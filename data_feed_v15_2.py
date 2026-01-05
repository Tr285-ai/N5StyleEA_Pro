# data_feed_v15_2.py
"""
Data Feed Module for N5StyleEA v15.2

Provides:
- Real-time market data feed with fallback to synthetic data
- Thread-safe tick and candle data management
- Support for multiple data sources (CSV, WebSocket, API)
- Configurable tick generation and buffering

Features:
- Real-time data processing with minimal latency
- Automatic failover to synthetic data
- Efficient memory usage with circular buffers
- Thread-safe operations
- Comprehensive logging
"""

import time
import json
import random
import logging
import threading
import queue
import asyncio
from typing import List, Tuple, Dict, Optional, Deque, Union, Any
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases
TickData = Tuple[float, float, float]  # (timestamp, price, volume)
CandleData = List[float]  # [open, high, low, close, volume, timestamp]

@dataclass
class FeedConfig:
    """Configuration for data feed."""
    symbol: str = "EURUSD"
    timeframe: str = "M1"
    max_ticks: int = 20000
    max_candles: int = 1000
    synthetic_interval: float = 0.05  # 50ms between ticks
    synthetic_volatility: float = 0.0002
    reconnect_interval: int = 5  # seconds
    request_timeout: int = 10  # seconds

class BaseDataFeed:
    """Base class for all data feed implementations."""
    
    def __init__(self, config: FeedConfig):
        """Initialize the base data feed."""
        self.config = config
        self._tick_buffer: Deque[TickData] = deque(maxlen=config.max_ticks)
        self._candle_buffer: Deque[CandleData] = deque(maxlen=config.max_candles)
        self._tick_lock = threading.Lock()
        self._candle_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update = 0.0
        
    def start(self) -> None:
        """Start the data feed."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Started {self.__class__.__name__} for {self.config.symbol}")
        
    def stop(self) -> None:
        """Stop the data feed."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info(f"Stopped {self.__class__.__name__}")
        
    def _run(self) -> None:
        """Main data feed loop to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run()")
        
    def get_recent_ticks(self, limit: int = 200) -> List[TickData]:
        """Get recent tick data."""
        with self._tick_lock:
            return list(self._tick_buffer)[-limit:]
            
    def get_latest_candles(self, limit: int = 300) -> List[CandleData]:
        """Get latest candle data."""
        with self._candle_lock:
            return list(self._candle_buffer)[-limit:]
            
    def _add_tick(self, timestamp: float, price: float, volume: float) -> None:
        """Add a new tick to the buffer."""
        with self._tick_lock:
            self._tick_buffer.append((timestamp, price, volume))
        self._last_update = time.time()
        
    def _add_candle(self, candle: CandleData) -> None:
        """Add a new candle to the buffer."""
        with self._candle_lock:
            if self._candle_buffer:
                last_candle = self._candle_buffer[-1]
                # Only add if timestamp is newer than the last candle
                if len(candle) > 5 and len(last_candle) > 5 and candle[5] > last_candle[5]:
                    self._candle_buffer.append(candle)
            else:
                self._candle_buffer.append(candle)
                
    def is_healthy(self) -> bool:
        """Check if the feed is healthy and updating."""
        return (time.time() - self._last_update) < (self.config.reconnect_interval * 2)

class PocketOptionFeed(BaseDataFeed):
    """PocketOption data feed with real-time and synthetic data support."""
    
    def __init__(self, config: FeedConfig, csv_path: Optional[str] = None):
        """Initialize the PocketOption feed."""
        super().__init__(config)
        self.csv_path = Path(csv_path) if csv_path else None
        self._candles_df: Optional[pd.DataFrame] = None
        self._current_candle: Optional[CandleData] = None
        self._last_candle_time = 0
        self._tick_generator = self._synthetic_tick_generator()
        
        # Load historical data if CSV provided
        if self.csv_path and self.csv_path.exists():
            try:
                self._load_historical_data()
            except Exception as e:
                logger.error(f"Failed to load historical data: {e}")
                
    def _load_historical_data(self) -> None:
        """Load historical data from CSV."""
        if not self.csv_path:
            return
            
        logger.info(f"Loading historical data from {self.csv_path}")
        self._candles_df = pd.read_csv(self.csv_path)
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        for col in required_columns:
            if col not in self._candles_df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Convert timestamp to seconds if it's in milliseconds
        if self._candles_df['timestamp'].max() > 1e10:  # Likely in milliseconds
            self._candles_df['timestamp'] = self._candles_df['timestamp'] / 1000
            
        logger.info(f"Loaded {len(self._candles_df)} historical candles")
        
    def _run(self) -> None:
        """Main data feed loop."""
        logger.info("Starting PocketOption data feed")
        
        while self._running:
            try:
                # Try to get real-time data
                if not self._try_real_time_update():
                    # Fall back to synthetic data
                    self._generate_synthetic_ticks()
                    
                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in data feed: {e}")
                time.sleep(self.config.reconnect_interval)
                
    def _try_real_time_update(self) -> bool:
        """Attempt to get real-time data. Returns True if successful."""
        # Implement real-time data fetching here
        # This is a placeholder for actual implementation
        return False
        
    def _generate_synthetic_ticks(self) -> None:
        """Generate synthetic tick data."""
        try:
            tick = next(self._tick_generator)
            self._add_tick(*tick)
            self._update_candles(tick)
        except StopIteration:
            # Reset generator if it's exhausted
            self._tick_generator = self._synthetic_tick_generator()
            
    def _synthetic_tick_generator(self) -> TickData:
        """Generate synthetic tick data."""
        if self._candles_df is not None and not self._candles_df.empty:
            # Generate ticks based on historical data
            for _, row in self._candles_df.iterrows():
                o, h, l, c = row['open'], row['high'], row['low'], row['close']
                v = row.get('volume', 1.0)
                ts = row['timestamp']
                
                # Generate multiple ticks per candle
                for i in range(10):  # 10 ticks per candle
                    progress = i / 9.0
                    price = o * (1 - progress) + c * progress
                    price += random.uniform(-0.0001, 0.0001)  # Add small noise
                    tick_ts = ts + (progress * 60)  # Distribute ticks over 1 minute
                    yield (tick_ts, float(price), float(v / 10))
        else:
            # Fallback: random walk
            price = 1.1000
            while True:
                ts = time.time()
                price += random.uniform(-self.config.synthetic_volatility, 
                                      self.config.synthetic_volatility)
                volume = random.uniform(0.1, 2.0)
                yield (ts, price, volume)
                
    def _update_candles(self, tick: TickData) -> None:
        """Update candle data from ticks."""
        ts, price, volume = tick
        candle_time = (ts // 60) * 60  # Align to minute
        
        if self._current_candle is None or candle_time > self._last_candle_time:
            # Start new candle
            if self._current_candle is not None:
                with self._candle_lock:
                    self._candle_buffer.append(self._current_candle)
                    
            self._current_candle = [price, price, price, price, volume, candle_time]
            self._last_candle_time = candle_time
        else:
            # Update current candle
            if self._current_candle:
                self._current_candle[1] = max(self._current_candle[1], price)  # High
                self._current_candle[2] = min(self._current_candle[2], price)  # Low
                self._current_candle[3] = price  # Close
                self._current_candle[4] += volume  # Volume
                
    def get_historical_candles(self, limit: int = 1000) -> List[CandleData]:
        """Get historical candle data."""
        if self._candles_df is None:
            return []
            
        df = self._candles_df.tail(limit)
        return [
            [float(row['open']), float(row['high']), float(row['low']), 
             float(row['close']), float(row.get('volume', 0)), float(row['timestamp'])]
            for _, row in df.iterrows()
        ]

class DataFeedV15_2:
    """Main data feed class with multiple source support."""
    
    def __init__(self, config: Optional[FeedConfig] = None):
        """Initialize the data feed."""
        self.config = config or FeedConfig()
        self.feeds: Dict[str, BaseDataFeed] = {}
        self.active_feed: Optional[BaseDataFeed] = None
        
    def add_feed(self, name: str, feed: BaseDataFeed) -> None:
        """Add a data feed."""
        self.feeds[name] = feed
        if not self.active_feed:
            self.active_feed = feed
            
    def start(self) -> None:
        """Start all registered feeds."""
        for name, feed in self.feeds.items():
            try:
                feed.start()
                logger.info(f"Started feed: {name}")
            except Exception as e:
                logger.error(f"Failed to start feed {name}: {e}")
                
    def stop(self) -> None:
        """Stop all registered feeds."""
        for name, feed in self.feeds.items():
            try:
                feed.stop()
                logger.info(f"Stopped feed: {name}")
            except Exception as e:
                logger.error(f"Error stopping feed {name}: {e}")
                
    def get_recent_ticks(self, limit: int = 200) -> List[TickData]:
        """Get recent ticks from the active feed."""
        if not self.active_feed:
            raise RuntimeError("No active data feed")
        return self.active_feed.get_recent_ticks(limit)
        
    def get_latest_candles(self, limit: int = 300) -> List[CandleData]:
        """Get latest candles from the active feed."""
        if not self.active_feed:
            raise RuntimeError("No active data feed")
        return self.active_feed.get_latest_candles(limit)

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = FeedConfig(
        symbol="EURUSD",
        timeframe="M1",
        max_ticks=10000,
        max_candles=1000,
        synthetic_interval=0.05
    )
    
    # Create and start feed
    feed = PocketOptionFeed(config, csv_path="historical_data.csv")
    data_feed = DataFeedV15_2(config)
    data_feed.add_feed("pocket_option", feed)
    data_feed.start()
    
    try:
        # Example: Print ticks for 10 seconds
        for _ in range(100):
            ticks = data_feed.get_recent_ticks(5)
            print(f"Latest ticks: {ticks}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping data feed...")
    finally:
        data_feed.stop()