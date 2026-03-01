import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
import builtins
from dotenv import load_dotenv
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from performance_monitor import performance_monitor
try:
    from alembic.config import Config as AlembicConfig  # type: ignore
    from alembic import command as alembic_command  # type: ignore
except Exception:
    AlembicConfig = None  # type: ignore
    alembic_command = None  # type: ignore

from logging_json import log_json
from trading_bot import TradingBot
from trading_strategy import TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
try:
    os.makedirs("logs", exist_ok=True)
    retention_days = int(os.getenv("LOG_RETENTION_DAYS", "7"))
    fh = TimedRotatingFileHandler(
        filename=os.path.join("logs", "app.log"), when="midnight", backupCount=max(0, retention_days)
    )
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root = logging.getLogger()
    root.addHandler(fh)
except Exception:
    # Do not fail startup if file handler cannot be created
    pass
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
    try:
        log_json("app_start")
    except Exception:
        logger.exception("Failed to emit app_start structured log")
    # Initialize resources here: construct and start the TradingBot as a background task
    try:
        load_dotenv()
        # Optional DB migration on startup (opt-in via env)
        try:
            migrate_on_start = str(os.getenv("DB_MIGRATE_ON_START", "0")).lower() in {"1", "true", "yes", "on"}
            db_url = os.getenv("ORDERS_DB_URL") or os.getenv("DATABASE_URL")
        except Exception:
            migrate_on_start = False
            db_url = None
        if migrate_on_start and AlembicConfig is not None and alembic_command is not None and db_url:
            try:
                cfg = AlembicConfig("alembic.ini")
                cfg.set_main_option("sqlalchemy.url", str(db_url))
                alembic_command.upgrade(cfg, "head")
            except Exception:
                logger.exception("alembic upgrade head failed during startup")
        email_config = {
            'smtp_server': os.getenv('SMTP_SERVER'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email': os.getenv('EMAIL'),
            'password': os.getenv('EMAIL_PASSWORD'),
        }
        strategy = TradingStrategy(
            model_path=os.getenv('MODEL_PATH', 'models/ensemble_model'),
            input_shape=(30, 5),
            email_config=email_config,
            econ_calendar_api_key=os.getenv('ECONOMIC_CALENDAR_API_KEY'),
        )
        paper_mode = os.getenv('PAPER_TRADING', 'true').lower() in {'1', 'true', 'yes', 'y', 'on'}
        exchange_name = os.getenv('EXCHANGE_NAME', 'binance')
        timeframe = os.getenv('TIMEFRAME', '1m')
        ohlcv_limit = int(os.getenv('OHLCV_LIMIT', '100'))
        api_key = os.getenv('EXCHANGE_API_KEY')
        api_secret = os.getenv('EXCHANGE_API_SECRET')
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
        app.state.bot = bot
        app.state.bot_task = asyncio.create_task(bot.run())
        # Event loop lag background gauge
        try:
            app.state.loop_lag_task = asyncio.create_task(_loop_lag_task())
        except Exception:
            logger.exception("Failed to start loop_lag_task")
    except Exception as e:
        logger.exception(f"Failed to start TradingBot during startup: {e}")
        app.state.bot = None
        app.state.bot_task = None
    yield
    # Clean up resources here: stop the TradingBot task gracefully
    try:
        task = getattr(app.state, 'bot_task', None)
        if task is not None:
            task.cancel()
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                pass
            try:
                await task
            except asyncio.CancelledError:
                pass
    except Exception:
        logger.exception("Failed to stop TradingBot during shutdown")
    # Stop loop lag task
    try:
        ll = getattr(app.state, 'loop_lag_task', None)
        if ll is not None:
            ll.cancel()
            try:
                await ll
            except asyncio.CancelledError:
                pass
    except Exception:
        logger.exception("Failed to stop loop lag task during shutdown")
    logger.info("Shutting down application")
    try:
        log_json("app_shutdown")
    except Exception:
        logger.exception("Failed to emit app_shutdown structured log")

# Create FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)
manager = ConnectionManager()

def _get_admin_token() -> str | None:
    tok = os.getenv("ADMIN_TOKEN")
    if tok:
        return tok.strip()
    path = os.getenv("ADMIN_TOKEN_FILE")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None

def _enforce_admin(request: Request) -> None:
    tok = _get_admin_token()
    if not tok:
        return
    provided = request.headers.get("authorization") or request.headers.get("x-admin-token")
    if not provided:
        raise HTTPException(status_code=401, detail="Admin token required")
    if provided.lower().startswith("bearer "):
        provided = provided[7:]
    if provided != tok:
        raise HTTPException(status_code=403, detail="Invalid admin token")

async def _loop_lag_task() -> None:
    try:
        interval = float(os.getenv("EVENT_LOOP_LAG_INTERVAL_SEC", "1.0"))
    except Exception:
        interval = 1.0
    last = time.perf_counter()
    while True:
        try:
            await asyncio.sleep(interval)
            now = time.perf_counter()
            elapsed = now - last
            lag_ms = max(0.0, (elapsed - interval) * 1000.0)
            try:
                performance_monitor.set_gauge("event_loop_lag_ms", float(lag_ms))
            except Exception:
                logger.exception("Failed to set event_loop_lag_ms gauge")
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("loop_lag_task error")
        finally:
            last = time.perf_counter()

# CORS middleware configuration
_allowed = os.getenv("ALLOWED_ORIGINS")
_origins = [o.strip() for o in _allowed.split(",") if o.strip()] if _allowed else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
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
async def metrics(request: Request):
    """Prometheus metrics endpoint."""
    _enforce_admin(request)
    try:
        from performance_monitor import performance_monitor as _pm
        monitor = getattr(builtins, "_N5_PERFMON_SINGLETON", _pm)
        try:
            monitor.set_gauge("app_heartbeat", float(time.time()))
        except Exception:
            logger.exception("Failed to set app_heartbeat gauge")
        body = monitor.render_prometheus()
    except Exception:
        body = ""
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/health/live")
async def health_live() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/health/ready")
async def health_ready() -> Dict[str, Any]:
    bot = getattr(app.state, "bot", None)
    if bot is None:
        try:
            log_json("health_ready", ready=False, reason="bot_not_initialized")
        except Exception:
            logger.exception("Failed to emit health_ready structured log")
        return {"ready": False, "reason": "bot_not_initialized"}

    try:
        ex_name = getattr(bot, "exchange_name", None)
        executor = getattr(bot, "executor", None)
        connected = bool(getattr(executor, "connected", False)) if executor else False
        try:
            log_json(
                "health_ready",
                ready=bool(connected),
                exchange=ex_name,
                connected=bool(connected),
            )
        except Exception:
            logger.exception("Failed to emit health_ready structured log")
        return {"ready": bool(connected), "exchange": ex_name, "connected": connected}
    except Exception as e:
        try:
            log_json("health_ready", ready=False, reason=str(e))
        except Exception:
            logger.exception("Failed to emit health_ready structured log")
        return {"ready": False, "reason": str(e)}

@app.get("/status")
async def status_snapshot() -> Dict[str, Any]:
    bot = getattr(app.state, "bot", None)
    executor = getattr(bot, "executor", None) if bot else None
    exchange_name = getattr(bot, "exchange_name", None) if bot else None
    paper = bool(getattr(executor, "demo_mode", True)) if executor else None

    breaker = None
    try:
        ex = getattr(executor, "_exchange_executor", None) if executor else None
        if ex is not None and hasattr(ex, "breaker_snapshot"):
            breaker = ex.breaker_snapshot()
    except Exception as e:
        logger.error(f"Error getting breaker snapshot: {e}")
        breaker = None

    try:
        from performance_monitor import performance_monitor

        connected = bool(getattr(executor, "connected", False)) if executor else False
        performance_monitor.set_gauge("bot_running", 1.0 if bool(getattr(bot, "running", False)) else 0.0)
        performance_monitor.set_gauge("executor_connected", 1.0 if connected else 0.0)
        if breaker and isinstance(breaker, dict):
            performance_monitor.set_gauge("breaker_open", 1.0 if bool(breaker.get("open")) else 0.0)
            try:
                performance_monitor.set_gauge("breaker_fail_count", float(breaker.get("fail_count", 0) or 0))
            except Exception as e:
                logger.error(f"Failed to set breaker_fail_count gauge: {e}")
            try:
                performance_monitor.set_gauge(
                    "breaker_open_remaining_sec",
                    float(breaker.get("open_remaining_sec", 0.0) or 0.0),
                )
            except Exception:
                logger.exception("Failed to set breaker_open_remaining_sec gauge")
    except Exception:
        logger.exception("Failed to update status gauges")

    return {
        "exchange": exchange_name,
        "paper": paper,
        "breaker": breaker,
        "bot_running": bool(getattr(bot, "running", False)) if bot else False,
        "use_ws_env": str(os.getenv("USE_WS", "")).lower(),
        "use_event_driven_env": str(os.getenv("USE_EVENT_DRIVEN", "")).lower(),
        "streamer_present": bool(getattr(bot, "_streamer", None) is not None) if bot else False,
    }

@app.get("/risk/limits")
async def risk_limits(request: Request) -> Dict[str, Any]:
    _enforce_admin(request)
    bot = getattr(app.state, "bot", None)
    executor = getattr(bot, "executor", None) if bot else None
    rm = getattr(executor, "risk_manager", None) if executor else None
    sl = dict(getattr(rm, "symbol_limits", {}) or {})
    return {"symbol_limits": sl}

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
        app.state.bot = bot
        await bot.run()

    except Exception as e:
        logger.error(f"Error in trading bot: {e}")
        raise

def start_uvicorn():
    """Start the FastAPI server with uvicorn."""
    uvicorn.run(
        app,
        host=os.getenv('HOST', '127.0.0.1'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('RELOAD', 'false').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'info')
    )

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        import tensorflow as tf  # type: ignore
        tf.get_logger().setLevel('ERROR')
    except Exception:
        logger.info("TensorFlow not available; continuing without adjusting TF logging")

    # Start the application (FastAPI lifespan will manage the bot task)
    try:
        try:
            if os.getenv('UVLOOP_ENABLE', '').lower() in {'1', 'true', 'yes', 'on'}:
                import uvloop  # type: ignore
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                logger.info("uvloop enabled")
        except Exception:
            logger.info("uvloop not available; continuing with default event loop")
        uvicorn.run(
            app,
            host=os.getenv('HOST', '127.0.0.1'),
            port=int(os.getenv('PORT', 8000)),
            reload=os.getenv('RELOAD', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'info'),
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
