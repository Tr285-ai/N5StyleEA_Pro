# auto_trader_updated.py
"""
N5StyleEA_Pro v15.2 - Auto Trader
Handles trade execution and position management
"""
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrader:
    def __init__(self, broker_client, market_data, model_registry, config):
        """Initialize the auto trader"""
        self.broker = broker_client
        self.market_data = market_data
        self.model_registry = model_registry
        self.config = config
        self.running = False
        self.active_trades = []
        self.trade_history = []
        
        # Event callbacks
        self.on_trade_executed = None
        self.on_trade_closed = None
        self.on_error = None
        
        # Initialize state
        self.daily_pnl = 0.0
        self.last_trade_time = None
        
    def execute_trade(self, signal):
        """Execute a trade based on signal"""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return None
                
            # Check if we can trade
            if not self._can_trade(signal):
                return None
                
            # Prepare order
            order = self._prepare_order(signal)
            
            # Execute order
            result = self.broker.create_order(order)
            
            if result.get('status') == 'FILLED':
                # Create trade record
                trade = {
                    'id': result['order_id'],
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'entry': result['fill_price'],
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'size': signal.size,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'OPEN',
                    'metadata': {
                        'signal': signal.to_dict(),
                        'order': result
                    }
                }
                
                # Add to active trades
                self.active_trades.append(trade)
                self.trade_history.append(trade)
                self.last_trade_time = datetime.utcnow()
                
                # Trigger callback
                if self.on_trade_executed:
                    self.on_trade_executed(trade)
                    
                logger.info(f"Trade executed: {trade}")
                return trade
                
        except Exception as e:
            error_msg = f"Error executing trade: {e}"
            logger.error(error_msg, exc_info=True)
            if self.on_error:
                self.on_error(error_msg)
            return None

    def close_trade(self, trade, reason='MANUAL'):
        """Close an open trade"""
        try:
            # Prepare close order (opposite direction)
            close_direction = 'SELL' if trade['direction'] == 'BUY' else 'BUY'
            close_price = self.market_data.get_bid() if close_direction == 'SELL' else self.market_data.get_ask()
            
            # Create close order
            close_order = {
                'symbol': trade['symbol'],
                'direction': close_direction,
                'type': 'MARKET',
                'size': trade['size'],
                'reduce_only': True,
                'client_order_id': f"close_{trade['id']}"
            }
            
            # Execute close order
            result = self.broker.create_order(close_order)
            
            if result.get('status') == 'FILLED':
                # Update trade
                trade.update({
                    'exit_price': result['fill_price'],
                    'exit_time': datetime.utcnow().isoformat(),
                    'status': 'CLOSED',
                    'close_reason': reason,
                    'pnl': self._calculate_pnl(trade, result['fill_price'])
                })
                
                # Remove from active trades
                if trade in self.active_trades:
                    self.active_trades.remove(trade)
                    
                # Update daily P&L
                self.daily_pnl += trade['pnl']
                
                # Trigger callback
                if self.on_trade_closed:
                    self.on_trade_closed(trade)
                    
                logger.info(f"Trade closed: {trade}")
                return True
                
        except Exception as e:
            error_msg = f"Error closing trade: {e}"
            logger.error(error_msg, exc_info=True)
            if self.on_error:
                self.on_error(error_msg)
            return False

    def _validate_signal(self, signal) -> bool:
        """Validate trading signal"""
        required_fields = ['symbol', 'direction', 'entry', 'stop_loss', 'take_profit']
        for field in required_fields:
            if not hasattr(signal, field) or getattr(signal, field) is None:
                logger.warning(f"Invalid signal: missing {field}")
                return False
        return True

    def _can_trade(self, signal) -> bool:
        """Check if we can execute a trade"""
        # Check if market is open
        if not self.market_data.is_market_open(signal.symbol):
            logger.warning(f"Market is closed for {signal.symbol}")
            return False
            
        # Check if we already have an open position
        if self._has_open_position(signal.symbol, signal.direction):
            logger.info(f"Already have an open position for {signal.symbol}")
            return False
            
        # Check daily loss limit
        if self.daily_pnl <= -self.config.get('max_daily_loss', 5.0):
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}%")
            return False
            
        # Check max open trades
        if len(self.active_trades) >= self.config.get('max_open_trades', 5):
            logger.warning(f"Max open trades reached: {len(self.active_trades)}")
            return False
            
        return True

    def _prepare_order(self, signal) -> Dict:
        """Prepare order dictionary from signal"""
        return {
            'symbol': signal.symbol,
            'direction': signal.direction,
            'type': 'MARKET',
            'size': signal.size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'leverage': self.config.get('leverage', 1),
            'client_order_id': f"{signal.symbol}_{int(time.time())}",
            'reduce_only': False
        }

    def _has_open_position(self, symbol: str, direction: str) -> bool:
        """Check if we already have an open position for the symbol/direction"""
        for trade in self.active_trades:
            if trade['symbol'] == symbol and trade['direction'] == direction:
                return True
        return False

    def _calculate_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate P&L for a trade"""
        if trade['direction'] == 'BUY':
            return (exit_price - trade['entry']) * trade['size']
        else:
            return (trade['entry'] - exit_price) * trade['size']

    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        return self.daily_pnl

    def start(self):
        """Start the auto trader"""
        if self.running:
            return
            
        logger.info("Starting auto trader...")
        self.running = True
        
        # Reset daily P&L at market open
        self._reset_daily_pnl()
        
        logger.info("Auto trader started")

    def stop(self):
        """Stop the auto trader"""
        if not self.running:
            return
            
        logger.info("Stopping auto trader...")
        self.running = False
        
        # Close all open positions
        for trade in self.active_trades[:]:
            self.close_trade(trade, 'SHUTDOWN')
            
        logger.info("Auto trader stopped")

    def _reset_daily_pnl(self):
        """Reset daily P&L at market open"""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")

    def get_active_trades(self) -> List[Dict]:
        """Get list of active trades"""
        return self.active_trades

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        return self.trade_history[-limit:] if self.trade_history else []

def main():
    """Test the auto trader"""
    from broker_client import BrokerClient
    from market_data import MarketDataFeed
    from model_registry import ModelRegistry
    
    # Initialize components
    broker = BrokerClient(api_key='demo', api_secret='demo', demo=True)
    market_data = MarketDataFeed(symbols=['EURUSD'], timeframe='M1', broker_client=broker)
    model_registry = ModelRegistry(registry_path='model_registry')
    
    # Initialize auto trader
    trader = AutoTrader(
        broker_client=broker,
        market_data=market_data,
        model_registry=model_registry,
        config={
            'max_open_trades': 5,
            'max_daily_loss': 5.0,
            'leverage': 1
        }
    )
    
    try:
        # Start components
        market_data.start()
        trader.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        trader.stop()
        market_data.stop()

if __name__ == "__main__":
    main()