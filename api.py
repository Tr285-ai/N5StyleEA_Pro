from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path
import logging
from typing import Dict, Any

from ..core.websocket.pocket_ws import WebSocketManager

logger = logging.getLogger('api')

class WebServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.ws_manager = WebSocketManager()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        # Serve static files
        static_dir = Path(__file__).parent / 'static'
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.ws_manager._handle_connection(websocket, None)
            
        # Root endpoint serves the dashboard
        @self.app.get("/")
        async def get_dashboard():
            return FileResponse(static_dir / 'index.html')
            
    async def start(self):
        """Start the web server"""
        await self.ws_manager.start()
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def stop(self):
        """Stop the web server"""
        await self.ws_manager.stop()

# Singleton instance
web_server = WebServer()

async def start_web_server():
    """Start the web server (to be called from main)"""
    await web_server.start()