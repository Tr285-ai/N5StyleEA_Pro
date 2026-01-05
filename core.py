# c:\N5StyleEA_v15\core.py
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system class that handles the trading logic."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trading system with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_running = False
        self.data = None
        self.strategy = None
        self.exchange = None
        self.indicators = {}
        
    async def start(self):
        """Start the trading system."""
        if self.is_running:
            logger.warning("Trading system is already running")
            return
            
        self.is_running = True
        logger.info("Starting trading system...")
        
        try:
            # Initialize components
            await self.initialize()
            
            # Main trading loop
            while self.is_running:
                try:
                    # Fetch and process market data
                    await self.update_market_data()
                    
                    # Generate trading signals
                    signals = await self.generate_signals()
                    
                    # Execute trades based on signals
                    if signals:
                        await self.execute_trades(signals)
                        
                    # Sleep for the configured interval
                    await asyncio.sleep(self.config.get('interval', 60))
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(5)  # Prevent tight loop on errors
                    
        except asyncio.CancelledError:
            logger.info("Trading system stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in trading system: {e}")
            raise
        finally:
            await self.cleanup()
            
    async def stop(self):
        """Stop the trading system."""
        logger.info("Stopping trading system...")
        self.is_running = False
        
    async def initialize(self):
        """Initialize trading system components."""
        logger.info("Initializing trading system components...")
        
        # Initialize exchange connection
        await self.initialize_exchange()
        
        # Load historical data
        await self.load_historical_data()
        
        # Initialize strategy
        self.initialize_strategy()
        
    async def initialize_exchange(self):
        """Initialize the exchange connection."""
        exchange_name = self.config.get('exchange', 'binance')
        logger.info(f"Initializing {exchange_name} exchange connection...")
        
        # In a real implementation, this would connect to the actual exchange
        # For now, we'll just simulate it
        self.exchange = {
            'name': exchange_name,
            'connected': True
        }
        
    async def load_historical_data(self):
        """Load historical market data."""
        data_source = self.config.get('data_source')
        
        if data_source and isinstance(data_source, str) and data_source.endswith('.csv'):
            try:
                self.data = pd.read_csv(data_source)
                logger.info(f"Loaded historical data from {data_source}")
            except Exception as e:
                logger.error(f"Failed to load historical data: {e}")
                raise
        else:
            # Generate sample data if no file is provided
            logger.warning("No data source provided, generating sample data")
            self.generate_sample_data()
            
    def generate_sample_data(self):
        """Generate sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.cumsum(np.random.randn(100) * 0.01 + 0.001) + 100
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.rand(100) * 0.5,
            'low': prices - np.random.rand(100) * 0.5,
            'close': prices + np.random.randn(100) * 0.1,
            'volume': np.random.randint(100, 1000, size=100)
        })
        
    def initialize_strategy(self):
        """Initialize the trading strategy."""
        strategy_name = self.config.get('strategy', 'mean_reversion')
        logger.info(f"Initializing {strategy_name} strategy...")
        
        if strategy_name == 'mean_reversion':
            self.strategy = MeanReversionStrategy()
        elif strategy_name == 'trend_following':
            self.strategy = TrendFollowingStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
    async def update_market_data(self):
        """Fetch and update market data."""
        if self.exchange and self.exchange.get('connected'):
            # In a real implementation, this would fetch data from the exchange
            logger.debug("Updating market data...")
            # Simulate data update
            await asyncio.sleep(0.1)
        else:
            logger.warning("Exchange not connected, using existing data")
            
    async def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on the current market data."""
        if self.data is None or self.data.empty:
            logger.warning("No data available for signal generation")
            return []
            
        if self.strategy is None:
            logger.error("No strategy initialized")
            return []
            
        try:
            # Calculate indicators
            self.calculate_indicators()
            
            # Generate signals using the selected strategy
            signals = self.strategy.generate_signals(self.data, self.indicators)
            
            logger.debug(f"Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
            
    def calculate_indicators(self):
        """Calculate technical indicators."""
        if self.data is None or self.data.empty:
            return
            
        df = self.data
        
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        self.indicators['rsi'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        self.indicators['macd'] = macd.macd()
        self.indicators['macd_signal'] = macd.macd_signal()
        self.indicators['macd_diff'] = macd.macd_diff()
        
        # ATR
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        self.indicators['atr'] = atr.average_true_range()
        
        # ADX
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        self.indicators['adx'] = adx.adx()
        
    async def execute_trades(self, signals: List[Dict[str, Any]]):
        """Execute trades based on the generated signals."""
        if not signals:
            return
            
        logger.info(f"Executing {len(signals)} trades...")
        
        for signal in signals:
            try:
                if signal['action'] == 'buy':
                    await self.execute_buy(signal)
                elif signal['action'] == 'sell':
                    await self.execute_sell(signal)
            except Exception as e:
                logger.error(f"Error executing trade: {e}")
                
    async def execute_buy(self, signal: Dict[str, Any]):
        """Execute a buy order."""
        logger.info(f"Executing BUY order: {signal}")
        # In a real implementation, this would place an order with the exchange
        await asyncio.sleep(0.1)  # Simulate network delay
        
    async def execute_sell(self, signal: Dict[str, Any]):
        """Execute a sell order."""
        logger.info(f"Executing SELL order: {signal}")
        # In a real implementation, this would place an order with the exchange
        await asyncio.sleep(0.1)  # Simulate network delay
        
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.is_running = False
        # Close exchange connection if needed
        if self.exchange and self.exchange.get('connected'):
            logger.info("Closing exchange connection...")
            self.exchange['connected'] = False

class MeanReversionStrategy:
    """Mean reversion trading strategy."""
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on mean reversion strategy."""
        signals = []
        
        if 'rsi' not in indicators or len(indicators['rsi']) < 2:
            return signals
            
        current_rsi = indicators['rsi'].iloc[-1]
        previous_rsi = indicators['rsi'].iloc[-2]
        
        # Oversold condition
        if current_rsi < 30 and previous_rsi >= 30:
            signals.append({
                'action': 'buy',
                'symbol': 'BTC/USDT',
                'price': data['close'].iloc[-1],
                'timestamp': datetime.now().isoformat(),
                'reason': 'RSI crossed above oversold (30)'
            })
            
        # Overbought condition
        elif current_rsi > 70 and previous_rsi <= 70:
            signals.append({
                'action': 'sell',
                'symbol': 'BTC/USDT',
                'price': data['close'].iloc[-1],
                'timestamp': datetime.now().isoformat(),
                'reason': 'RSI crossed below overbought (70)'
            })
            
        return signals

class TrendFollowingStrategy:
    """Trend following trading strategy."""
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on trend following strategy."""
        signals = []
        
        if 'macd' not in indicators or 'macd_signal' not in indicators:
            return signals
            
        current_macd = indicators['macd'].iloc[-1]
        current_signal = indicators['macd_signal'].iloc[-1]
        previous_macd = indicators['macd'].iloc[-2]
        previous_signal = indicators['macd_signal'].iloc[-2]
        
        # Bullish crossover
        if previous_macd < previous_signal and current_macd > current_signal:
            signals.append({
                'action': 'buy',
                'symbol': 'BTC/USDT',
                'price': data['close'].iloc[-1],
                'timestamp': datetime.now().isoformat(),
                'reason': 'MACD bullish crossover'
            })
            
        # Bearish crossover
        elif previous_macd > previous_signal and current_macd < current_signal:
            signals.append({
                'action': 'sell',
                'symbol': 'BTC/USDT',
                'price': data['close'].iloc[-1],
                'timestamp': datetime.now().isoformat(),
                'reason': 'MACD bearish crossover'
            })
            
        return signals