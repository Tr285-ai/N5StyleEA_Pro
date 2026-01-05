# orchestrator_v15_2.py
"""
N5StyleEA_Pro v15.2 - Main Orchestrator
Handles the core trading logic and coordinates between components
"""
import os
import json
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Data class for trading signals"""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: str
    signal_type: str = 'REGULAR'
    expiry: Optional[float] = None
    size: float = 1.0
    source: str = 'orchestrator'
    metadata: dict = None

    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return asdict(self)

class Orchestrator:
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the orchestrator with configuration"""
        self.config = self._load_config(config_path)
        self.running = False
        self.signals = []
        self.active_trades = []
        self.market_state = {}
        
        # Initialize components
        self._init_components()
        
        # Initialize state
        self.last_candle_time = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _init_components(self):
        """Initialize all required components"""
        # Import components
        from broker_client import BrokerClient
        from market_data import MarketDataFeed
        from auto_trader_updated import AutoTrader
        from model_registry import ModelRegistry
        from self_learning_system import SelfLearningSystem

        # Initialize broker client
        self.broker = BrokerClient(
            api_key=self.config.get('broker_api_key'),
            api_secret=self.config.get('broker_api_secret'),
            demo=self.config.get('demo_mode', True)
        )
        
        # Initialize market data feed
        self.market_data = MarketDataFeed(
            symbols=self.config.get('symbols', ['EURUSD']),
            timeframe=self.config.get('timeframe', 'M1'),
            broker_client=self.broker
        )
        
        # Initialize model registry
        self.model_registry = ModelRegistry(
            registry_path=self.config.get('model_registry_path', 'model_registry')
        )
        
        # Initialize self-learning system
        self.self_learning = SelfLearningSystem(
            model_registry=self.model_registry,
            broker_client=self.broker,
            config=self.config.get('self_learning', {})
        )
        
        # Initialize auto trader
        self.auto_trader = AutoTrader(
            broker_client=self.broker,
            market_data=self.market_data,
            model_registry=self.model_registry,
            config=self.config.get('auto_trader', {})
        )
        
        # Register callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register callbacks for market data events"""
        self.market_data.on_candle = self._on_new_candle
        self.market_data.on_tick = self._on_tick
        self.auto_trader.on_trade_executed = self._on_trade_executed

    def _on_new_candle(self, symbol: str, candle: Dict[str, Any]):
        """Handle new candle data"""
        try:
            logger.info(f"New candle for {symbol}: {candle}")
            
            # Update market state
            self._update_market_state(symbol, candle)
            
            # Generate signals
            signals = self._generate_signals(symbol, candle)
            
            # Process signals
            for signal in signals:
                self._process_signal(signal)
                
        except Exception as e:
            logger.error(f"Error processing candle for {symbol}: {e}", exc_info=True)

    def _on_tick(self, symbol: str, tick: Dict[str, Any]):
        """Handle new tick data"""
        try:
            # Update micro-predictor if available
            if hasattr(self, 'micro_predictor'):
                self.micro_predictor.update(tick)
                
            # Update open positions
            self._update_positions(symbol, tick)
            
        except Exception as e:
            logger.error(f"Error processing tick for {symbol}: {e}", exc_info=True)

    def _on_trade_executed(self, trade: Dict[str, Any]):
        """Handle executed trade"""
        try:
            logger.info(f"Trade executed: {trade}")
            self.active_trades.append(trade)
            
            # Record trade for self-learning
            self.self_learning.record_trade(trade)
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}", exc_info=True)

    def _update_market_state(self, symbol: str, candle: Dict[str, Any]):
        """Update internal market state"""
        if symbol not in self.market_state:
            self.market_state[symbol] = {}
            
        self.market_state[symbol].update({
            'last_close': candle['close'],
            'last_high': candle['high'],
            'last_low': candle['low'],
            'last_volume': candle['volume'],
            'last_update': datetime.utcnow().isoformat()
        })

    def _generate_signals(self, symbol: str, candle: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on market data"""
        signals = []
        
        try:
            # Get model predictions
            predictions = self.model_registry.predict(symbol, candle)
            
            # Generate signals based on predictions
            for model_name, prediction in predictions.items():
                if prediction['confidence'] >= self.config.get('min_confidence', 0.7):
                    signal = Signal(
                        symbol=symbol,
                        direction=prediction['direction'],
                        entry=candle['close'],
                        stop_loss=prediction.get('stop_loss'),
                        take_profit=prediction.get('take_profit'),
                        confidence=prediction['confidence'],
                        timestamp=datetime.utcnow().isoformat(),
                        source=model_name,
                        metadata=prediction.get('metadata', {})
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
            
        return signals

    def _process_signal(self, signal: Signal):
        """Process a trading signal"""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return
                
            # Check risk parameters
            if not self._check_risk_parameters(signal):
                return
                
            # Execute trade
            self.auto_trader.execute_trade(signal)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}", exc_info=True)

    def _validate_signal(self, signal: Signal) -> bool:
        """Validate trading signal"""
        required_fields = ['symbol', 'direction', 'entry', 'stop_loss', 'take_profit']
        for field in required_fields:
            if not getattr(signal, field, None):
                logger.warning(f"Invalid signal: missing {field}")
                return False
        return True

    def _check_risk_parameters(self, signal: Signal) -> bool:
        """Check if trade meets risk parameters"""
        # Check daily loss limit
        daily_pnl = self.auto_trader.get_daily_pnl()
        if daily_pnl <= -self.config.get('max_daily_loss', 5.0):
            logger.warning(f"Daily loss limit reached: {daily_pnl}%")
            return False
            
        # Check max open trades
        if len(self.active_trades) >= self.config.get('max_open_trades', 5):
            logger.warning(f"Max open trades reached: {len(self.active_trades)}")
            return False
            
        return True

    def _update_positions(self, symbol: str, tick: Dict[str, Any]):
        """Update open positions based on market data"""
        for trade in self.active_trades[:]:
            if trade['symbol'] == symbol:
                # Update trade P&L
                trade['current_price'] = tick['bid'] if trade['direction'] == 'BUY' else tick['ask']
                trade['pnl'] = self._calculate_pnl(trade)
                
                # Check for exit conditions
                if self._should_exit_trade(trade, tick):
                    self.auto_trader.close_trade(trade)
                    self.active_trades.remove(trade)

    def _should_exit_trade(self, trade: Dict, tick: Dict) -> bool:
        """Check if trade should be exited"""
        current_price = tick['bid'] if trade['direction'] == 'BUY' else tick['ask']
        
        # Check stop loss
        if (trade['direction'] == 'BUY' and current_price <= trade['stop_loss']) or \
           (trade['direction'] == 'SELL' and current_price >= trade['stop_loss']):
            logger.info(f"Stop loss hit for trade {trade['id']}")
            return True
            
        # Check take profit
        if (trade['direction'] == 'BUY' and current_price >= trade['take_profit']) or \
           (trade['direction'] == 'SELL' and current_price <= trade['take_profit']):
            logger.info(f"Take profit hit for trade {trade['id']}")
            return True
            
        return False

    def _calculate_pnl(self, trade: Dict) -> float:
        """Calculate profit/loss for a trade"""
        if 'current_price' not in trade:
            return 0.0
            
        if trade['direction'] == 'BUY':
            return (trade['current_price'] - trade['entry']) * trade['size']
        else:
            return (trade['entry'] - trade['current_price']) * trade['size']

    def start(self):
        """Start the orchestrator"""
        if self.running:
            return
            
        logger.info("Starting orchestrator...")
        self.running = True
        
        # Start market data feed
        self.market_data.start()
        
        # Start auto trader
        self.auto_trader.start()
        
        logger.info("Orchestrator started successfully")

    def stop(self):
        """Stop the orchestrator"""
        if not self.running:
            return
            
        logger.info("Stopping orchestrator...")
        self.running = False
        
        # Stop all components
        self.market_data.stop()
        self.auto_trader.stop()
        
        logger.info("Orchestrator stopped")

def main():
    """Main entry point"""
    try:
        orchestrator = Orchestrator()
        orchestrator.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        orchestrator.stop()

if __name__ == "__main__":
    main()