# web/websocket_manager.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.market_data_clients: List[WebSocket] = []
        self.trade_clients: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, client_type: str = "default"):
        await websocket.accept()
        if client_type == "market_data":
            self.market_data_clients.append(websocket)
        elif client_type == "trades":
            self.trade_clients.append(websocket)
        else:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket, client_type: str = "default"):
        if client_type == "market_data" and websocket in self.market_data_clients:
            self.market_data_clients.remove(websocket)
        elif client_type == "trades" and websocket in self.trade_clients:
            self.trade_clients.remove(websocket)
        elif websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str, client_type: str = "default"):
        clients = self._get_clients(client_type)
        for connection in clients:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection, client_type)

    def _get_clients(self, client_type: str):
        if client_type == "market_data":
            return self.market_data_clients
        elif client_type == "trades":
            return self.trade_clients
        return self.active_connections

# Initialize WebSocket manager
manager = ConnectionManager()

# WebSocket endpoints
async def websocket_endpoint(websocket: WebSocket, client_type: str = "default"):
    await manager.connect(websocket, client_type)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(10)
            await websocket.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_type)