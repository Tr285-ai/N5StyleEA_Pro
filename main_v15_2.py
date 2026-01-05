# main_v15_2.py
"""
N5StyleEA v15.2 - Main Application

This is the main entry point for the N5StyleEA trading system.
It initializes all components and manages the trading loop.

Author: N5StyleEA Team
Version: 15.2.1
"""

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base_broker import BaseBroker, Order, OrderSide, OrderType
from .pocketoption import PocketOptionBroker
from .websocket_server import WebSocketServer
from .train_micro_models import MicroModelTrainer, ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('n5style_ea.log')
    ]
)
logger = logging.getLogger('n5style_ea')

class N5StyleEA:
    """Main application class for N5StyleEA trading system."""
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize the N5StyleEA application.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.broker: Optional[BaseBroker] = None
        self.ws_server: Optional[WebSocketServer] = None
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Initialize components
        self._init_broker()
        self._init_websocket_server()
        
        # Load models
        self.models: Dict[str, Any] = {}
        self._load_models()
        
        # Trading state
        self.positions: Dict[str, Any] = {}
        self.orders: Dict[str, Dict] = {}
        self.account: Dict[str, Any] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            sys.exit(1)

    def _init_broker(self) -> None:
        """Initialize the broker connection."""
        broker_config = self.config.get('broker', {})
        broker_type = broker_config.get('type', 'pocketoption')
        
        if broker_type.lower() == 'pocketoption':
            self.broker = PocketOptionBroker(
                email=broker_config.get('email'),
                password=broker_config.get('password'),
                is_demo=broker_config.get('demo', True)
            )
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")

    def _init_websocket_server(self) -> None:
        """Initialize the WebSocket server."""
        ws_config = self.config.get('websocket', {})
        self.ws_server = WebSocketServer(
            host=ws_config.get('host', '0.0.0.0'),
            port=ws_config.get('port', 8765)
        )
        
        # Register authentication handler
        self.ws_server.authenticate = self._authenticate

    def _load_models(self) -> None:
        """Load trained models."""
        models_config = self.config.get('models', {})
        models_dir = Path(models_config.get('directory', 'models'))
        
        for model_name, model_config in models_config.get('list', {}).items():
            model_path = models_dir / model_config['path']
            model_type = ModelType[model_config['type'].upper()]
            
            try:
                self.models[model_name] = MicroModelTrainer.load_model(
                    model_path=model_path,
                    model_type=model_type,
                    input_shape=tuple(model_config['input_shape'])
                )
                logger.info(f"Loaded model: {model_name} ({model_type.value})")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")

    async def start(self) -> None:
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system is already running")
            return
            
        self.running = True
        logger.info("Starting N5StyleEA...")
        
        try:
            # Connect to broker
            logger.info("Connecting to broker...")
            if not await self.broker.connect():
                raise ConnectionError("Failed to connect to broker")
                
            # Start WebSocket server
            logger.info("Starting WebSocket server...")
            await self.ws_server.start()
            
            # Start background tasks
            self.tasks = [
                asyncio.create_task(self._market_data_loop()),
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._monitor_positions()),
                asyncio.create_task(self._update_account_info())
            ]
            
            logger.info("N5StyleEA started successfully")
            
            # Keep the application running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}", exc_info=True)
            await self.stop()
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading system gracefully."""
        if not self.running:
            return
            
        logger.info("Stopping N5StyleEA...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Disconnect from broker
        if self.broker and self.broker.connected:
            await self.broker.disconnect()
            
        # Stop WebSocket server
        if self.ws_server:
            await self.ws_server.stop()
            
        logger.info("N5StyleEA stopped")

    async def _market_data_loop(self) -> None:
        """Background task for handling market data updates."""
        logger.info("Market data loop started")
        
        while self.running:
            try:
                # Get account info
                account = await self.broker.get_account_info()
                self.account = {
                    'equity': account.equity,
                    'balance': account.balance,
                    'margin_available': account.margin_available,
                    'margin_used': account.margin_used,
                    'leverage': account.leverage
                }
                
                # Update WebSocket clients
                if self.ws_server:
                    await self.ws_server.send_account_update(account)
                    
                # Get positions
                positions = await self.broker.get_positions()
                self.positions = {p.symbol: p for p in positions}
                
                # Update WebSocket clients
                for position in positions:
                    if self.ws_server:
                        await self.ws_server.send_position_update(position)
                
                # Sleep before next update
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in market data loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _trading_loop(self) -> None:
        """Background task for executing trading strategies."""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # Implement your trading strategy here
                # This is a placeholder - replace with your actual strategy
                
                # Example: Check for trading signals
                signals = await self._check_signals()
                
                # Process signals
        async def _trading_loop(self) -> None:
        """Background task for executing trading strategies."""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # Check for trading signals
                signals = await self._check_signals()
                await self._process_signals(signals)
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _monitor_positions(self) -> None:
        """Monitor open positions and manage risk."""
        logger.info("Position monitoring started")
        
        while self.running:
            try:
                # Check stop-loss and take-profit levels
                await self._check_position_limits()
                
                # Update position metrics
                await self._update_position_metrics()
                
                # Sleep before next check
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _update_account_info(self) -> None:
        """Periodically update account information."""
        logger.info("Account info update started")
        
        while self.running:
            try:
                if self.broker:
                    account = await self.broker.get_account_info()
                    self.account = {
                        'equity': account.equity,
                        'balance': account.balance,
                        'margin': account.margin_available,
                        'leverage': account.leverage
                    }
                    
                    # Update WebSocket clients
                    if self.ws_server:
                        await self.ws_server.send_account_update(account)
                
                # Update every 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error updating account info: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _check_signals(self) -> List[Dict[str, Any]]:
        """Check for trading signals from models."""
        signals = []
        
        try:
            # Get current market data
            symbols = self.config.get('trading', {}).get('symbols', [])
            
            for symbol in symbols:
                # Get recent candles
                candles = await self.broker.get_historical_data(
                    symbol=symbol,
                    timeframe='1m',
                    limit=100
                )
                
                # Generate features
                features = self._generate_features(candles)
                
                # Get predictions from models
                predictions = {}
                for model_name, model in self.models.items():
                    if model.is_ready():
                        prediction = model.predict(features)
                        predictions[model_name] = prediction
                
                # Generate signals based on predictions
                signal = self._generate_signal(predictions)
                if signal:
                    signals.append({
                        'symbol': symbol,
                        'signal': signal,
                        'timestamp': int(time.time() * 1000)
                    })
                    
        except Exception as e:
            logger.error(f"Error checking signals: {str(e)}")
            
        return signals

    async def _process_signals(self, signals: List[Dict[str, Any]]) -> None:
        """Process trading signals and execute orders."""
        for signal in signals:
            try:
                symbol = signal['symbol']
                signal_type = signal['signal']
                
                # Check if we already have an open position
                if symbol in self.positions:
                    continue
                
                # Execute trade based on signal
                order = await self._create_order(symbol, signal_type)
                if order:
                    await self.broker.place_order(order)
                    logger.info(f"Order placed: {order}")
                    
            except Exception as e:
                logger.error(f"Error processing signal: {str(e)}")

    async def _create_order(self, symbol: str, signal_type: str) -> Optional[Order]:
        """Create an order based on signal."""
        try:
            # Get current price
            price = await self.broker.get_current_price(symbol)
            
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(price)
            
            # Create order
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if signal_type == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position_size,
                price=price
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            return None

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management rules."""
        risk_per_trade = self.config.get('trading', {}).get('risk_per_trade', 0.01)
        stop_loss_pct = self.config.get('trading', {}).get('stop_loss_pct', 0.02)
        
        if not self.account or 'balance' not in self.account:
            return 0.0
            
        risk_amount = self.account['balance'] * risk_per_trade
        position_size = risk_amount / (price * stop_loss_pct)
        
        return round(position_size, 8)  # Round to 8 decimal places

    def _generate_features(self, candles: List[Dict]) -> Dict[str, Any]:
        """Generate features from candle data for model prediction."""
        # Implement your feature engineering here
        return {}

    def _generate_signal(self, predictions: Dict[str, Any]) -> Optional[str]:
        """Generate trading signal from model predictions."""
        # Implement your signal generation logic here
        return None

    async def _check_position_limits(self) -> None:
        """Check position limits and close if necessary."""
        # Implement position limit checks
        pass

    async def _update_position_metrics(self) -> None:
        """Update metrics for open positions."""
        # Implement position metrics updates
        pass

    def _authenticate(self, username: str, password: str) -> bool:
        """Authenticate WebSocket clients."""
        # Implement your authentication logic here
        return username == self.config.get('websocket', {}).get('username') and \
               password == self.config.get('websocket', {}).get('password')

async def main():
    """Main entry point."""
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Create and start the application
        app = N5StyleEA('config.json')
        await app.start()
        
        # Wait for stop event
        await stop_event.wait()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        if 'app' in locals():
            await app.stop()
        logger.info("Application stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)            