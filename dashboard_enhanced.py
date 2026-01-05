# web/dashboard_enhanced.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import random
from datetime import datetime
import uvicorn
from pathlib import Path

# Import our components
from .auth import get_current_active_user, User
from .websocket_manager import manager, websocket_endpoint
from ..exchanges.implementations import BinanceExchange, FTXExchange, KrakenExchange
from ..risk.ml_risk_model import MLRiskModel

app = FastAPI()

# Set up templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize components
exchanges = {
    'binance': BinanceExchange(),
    'ftx': FTXExchange(),
    'kraken': KrakenExchange()
}
risk_model = MLRiskModel()

# WebSocket endpoints
@app.websocket("/ws/market-data")
async def market_data_websocket(websocket: WebSocket):
    await websocket_endpoint(websocket, "market_data")

@app.websocket("/ws/trades")
async def trades_websocket(websocket: WebSocket):
    await websocket_endpoint(websocket, "trades")

# Background task for market data updates
async def broadcast_market_data():
    while True:
        try:
            # Simulate market data updates
            for exchange_name, exchange in exchanges.items():
                try:
                    ticker = await exchange.get_ticker("BTC/USDT")
                    await manager.broadcast(json.dumps({
                        "type": "market_data",
                        "exchange": exchange_name,
                        "data": ticker,
                        "timestamp": datetime.utcnow().isoformat()
                    }), "market_data")
                except Exception as e:
                    print(f"Error getting ticker from {exchange_name}: {e}")
            
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in broadcast_market_data: {e}")
            await asyncio.sleep(5)

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("dashboard_enhanced.html", {
        "request": request,
        "user": current_user
    })

@app.get("/api/risk-assessment")
async def get_risk_assessment():
    # In a real app, fetch actual market data
    market_data = pd.DataFrame({
        'close': [random.uniform(30000, 50000) for _ in range(100)],
        'volume': [random.uniform(1000, 10000) for _ in range(100)],
        'timestamp': [datetime.utcnow()] * 100
    })
    
    risk_score, feature_importance = risk_model.predict_risk(market_data)
    
    return {
        "risk_score": risk_score,
        "feature_importance": feature_importance,
        "timestamp": datetime.utcnow().isoformat()
    }

# Start the background task on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_market_data())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)