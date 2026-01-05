import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Set, Any
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger('trading.websocket')

class WebSocketManager:
    """
    WebSocket server for real-time communication with clients.
    Handles connections, message broadcasting, and client management.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """
        Initialize the WebSocket manager.
        
        Args:
            host: Host to bind the server to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self.is_running = False
        self._message_handlers = {}
        
    async def start(self) -> None:
        """Start the WebSocket server."""
        if self.is_running:
            logger.warning("WebSocket server is already running")
            return
            
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_running = True
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the WebSocket server and close all connections."""
        if not self.is_running:
            return
            
        logger.info("Stopping WebSocket server...")
        self.is_running = False
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[self._close_client(client) for client in list(self.clients)],
                return_exceptions=True
            )
            self.clients.clear()
            
        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        logger.info("WebSocket server stopped")
        
    async def _close_client(self, client: WebSocketServerProtocol) -> None:
        """Safely close a client connection."""
        try:
            if client.open:
                await client.close()
        except Exception as e:
            logger.error(f"Error closing client connection: {e}")
            
    async def _handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ) -> None:
        """Handle a new WebSocket connection."""
        remote_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New WebSocket connection from {remote_addr}")
        
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed: {remote_addr}")
        except Exception as e:
            logger.error(f"WebSocket error from {remote_addr}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {remote_addr}")
            
    async def _handle_message(
        self,
        websocket: WebSocketServerProtocol,
        message: str
    ) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self._message_handlers:
                await self._message_handlers[message_type](websocket, data)
            else:
                logger.warning(f"No handler for message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
    def register_handler(
        self,
        message_type: str,
        handler: Callable[[WebSocketServerProtocol, Dict], None]
    ) -> None:
        """
        Register a message handler for a specific message type.
        
        Args:
            message_type: The message type to handle
            handler: Coroutine function that handles the message
        """
        self._message_handlers[message_type] = handler
        
    async def broadcast(self, message: Any) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast (will be JSON-serialized)
        """
        if not self.clients:
            return
            
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
                
            await asyncio.gather(
                *[self._safe_send(client, message) for client in list(self.clients)],
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
            
    async def _safe_send(
        self,
        client: WebSocketServerProtocol,
        message: str
    ) -> None:
        """Safely send a message to a client."""
        try:
            if client.open:
                await client.send(message)
        except ConnectionClosed:
            self.clients.discard(client)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            self.clients.discard(client)