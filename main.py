import asyncio
import logging
import os
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import pandas as pd
import tensorflow as tf
from trading_strategy import TradingStrategy
from trading_bot import TradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:  # Create a copy of the list
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                self.disconnect(connection)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    logger.info("Starting application")
    # Initialize resources here
    yield
    # Clean up resources here
    logger.info("Shutting down application")

# Create FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)
manager = ConnectionManager()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MarketData(BaseModel):
    """Model for market data received via WebSocket."""
    symbol: str
    price: float
    volume: float
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "N5StyleEA Trading System"}



@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        from performance_monitor import performance_monitor
        body = performance_monitor.render_prometheus()
    except Exception:
        body = ""
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")
@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market data."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process the received data
            await manager.broadcast(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

async def run_trading_bot():
    """Initialize and run the trading bot."""
    # Load environment variables
    load_dotenv()
    
    # Email configuration
    email_config = {
        'smtp_server': os.getenv('SMTP_SERVER'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'email': os.getenv('EMAIL'),
        'password': os.getenv('EMAIL_PASSWORD')
    }
    
    try:
        # Initialize trading strategy
        strategy = TradingStrategy(
            model_path=os.getenv('MODEL_PATH', 'models/ensemble_model'),
            input_shape=(30, 5),  # Adjust based on your data
            email_config=email_config,
            econ_calendar_api_key=os.getenv('ECONOMIC_CALENDAR_API_KEY')
        )
        
        paper_mode = os.getenv('PAPER_TRADING', 'true').lower() in {'1', 'true', 'yes', 'y', 'on'}
        exchange_name = os.getenv('EXCHANGE_NAME', 'binance')
        timeframe = os.getenv('TIMEFRAME', '1m')
        ohlcv_limit = int(os.getenv('OHLCV_LIMIT', '100'))

        api_key = os.getenv('EXCHANGE_API_KEY')
        api_secret = os.getenv('EXCHANGE_API_SECRET')
        
        # Initialize trading bot
        bot = TradingBot(
            strategy=strategy,
            symbols=os.getenv('TRADING_SYMBOLS', 'EUR/USD,GBP/USD').split(','),
            initial_balance=float(os.getenv('INITIAL_BALANCE', 10000.0)),
            exchange_name=exchange_name,
            timeframe=timeframe,
            ohlcv_limit=ohlcv_limit,
            paper=paper_mode,
            api_key=api_key,
            api_secret=api_secret,
        )
        
        # Start trading
        logger.info("Starting trading bot...")
        await bot.run()
        
    except Exception as e:
        logger.error(f"Error in trading bot: {e}")
        raise

def start_uvicorn():
    """Start the FastAPI server with uvicorn."""
    uvicorn.run(
        "main:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('RELOAD', 'false').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'info')
    )

if __name__ == "__main__":
    # Set TensorFlow log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # Start the application
    try:
        import threading

        # Start web server in a separate thread
        web_thread = threading.Thread(target=start_uvicorn, daemon=True)
        web_thread.start()

        # Run the trading bot in the main thread
        asyncio.run(run_trading_bot())
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")