# websocket_server.py
"""
WebSocket Server

This module implements a WebSocket server for real-time communication
between the trading system and clients.

Author: N5StyleEA Team
Version: 15.2.1
"""

import asyncio
import json
import logging
import signal
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Set, Callable

import websockets
from websockets.exceptions import ConnectionClosed

from .base_broker import (
    Order, OrderStatus, Position, AccountInfo,
    OrderType, OrderSide, TimeInForce, PositionSide
)

logger = logging.getLogger(__name__)

class WebSocketServer:
    """WebSocket server for real-time trading updates."""
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 8765,
        ssl_context = None,
        max_connections: int = 100,
        ping_interval: int = 30,
        ping_timeout: int = 10
    ):
        """
        Initialize the WebSocket server.
        
        Args:
            host: Host to bind the server to
            port: Port to listen on
            ssl_context: SSL context for secure connections
            max_connections: Maximum number of concurrent connections
            ping_interval: Interval for sending ping messages (seconds)
            ping_timeout: Timeout for ping responses (seconds)
        """
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.max_connections = max_connections
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        self.server = None
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> set of client_ids
        self.handlers = {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'ping': self._handle_ping,
            'auth': self._handle_auth
        }
        
        # Statistics
        self.stats = {
            'connections': 0,
            'messages_received': 0,
            'messages_sent': 0,
            'errors': 0
        }
        
        # Authentication callback (set by the application)
        self.authenticate: Optional[Callable[[str, str], bool]] = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self.server is not None:
            raise RuntimeError("Server is already running")
            
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            max_size=2**25,  # 32MB max message size
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            close_timeout=0
        )
        
        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)
        
        logger.info("WebSocket server started")

    async def stop(self) -> None:
        """Stop the WebSocket server gracefully."""
        if self.server is None:
            return
            
        logger.info("Shutting down WebSocket server...")
        
        # Close all client connections
        if self.clients:
            logger.info(f"Closing {len(self.clients)} client connections...")
            await asyncio.gather(*[
                self._close_client(websocket, 1001, "Server shutdown")
                for websocket in self.clients.values()
            ], return_exceptions=True)
        
        # Stop the server
        self.server.close()
        await self.server.wait_closed()
        self.server = None
        
        logger.info("WebSocket server stopped")

    async def _handle_connection(self, websocket, path: str) -> None:
        """Handle a new WebSocket connection."""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        self.stats['connections'] += 1
        
        logger.info(f"Client connected: {client_id} (Total: {len(self.clients)})")
        
        try:
            async for message in websocket:
                try:
                    await self._process_message(client_id, message)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}", exc_info=True)
                    self.stats['errors'] += 1
                    await self._send_error(client_id, str(e))
        except ConnectionClosed:
            logger.debug(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}", exc_info=True)
            self.stats['errors'] += 1
        finally:
            # Clean up
            await self._cleanup_client(client_id)
            logger.info(f"Client disconnected: {client_id} (Remaining: {len(self.clients)})")

    async def _process_message(self, client_id: str, message: str) -> None:
        """Process an incoming message from a client."""
        try:
            self.stats['messages_received'] += 1
            data = json.loads(message)
            
            # Validate message format
            if not isinstance(data, dict) or 'type' not in data:
                raise ValueError("Invalid message format: missing 'type' field")
                
            # Route to appropriate handler
            handler = self.handlers.get(data['type'])
            if not handler:
                raise ValueError(f"Unknown message type: {data['type']}")
                
            await handler(client_id, data)
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            raise

    async def _handle_subscribe(self, client_id: str, data: dict) -> None:
        """Handle subscription request."""
        topic = data.get('topic')
        if not topic:
            raise ValueError("Missing 'topic' in subscription")
            
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
            
        self.subscriptions[topic].add(client_id)
        await self._send_message(client_id, {
            'type': 'subscription',
            'topic': topic,
            'status': 'subscribed'
        })

    async def _handle_unsubscribe(self, client_id: str, data: dict) -> None:
        """Handle unsubscription request."""
        topic = data.get('topic')
        if not topic:
            raise ValueError("Missing 'topic' in unsubscription")
            
        if topic in self.subscriptions and client_id in self.subscriptions[topic]:
            self.subscriptions[topic].remove(client_id)
            
        await self._send_message(client_id, {
            'type': 'subscription',
            'topic': topic,
            'status': 'unsubscribed'
        })

    async def _handle_ping(self, client_id: str, data: dict) -> None:
        """Handle ping message."""
        await self._send_message(client_id, {
            'type': 'pong',
            'timestamp': data.get('timestamp')
        })

    async def _handle_auth(self, client_id: str, data: dict) -> None:
        """Handle authentication request."""
        if not self.authenticate:
            raise RuntimeError("Authentication not configured")
            
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            raise ValueError("Missing username or password")
            
        authenticated = await asyncio.get_event_loop().run_in_executor(
            None, self.authenticate, username, password
        )
        
        if not authenticated:
            raise PermissionError("Authentication failed")
            
        await self._send_message(client_id, {
            'type': 'auth',
            'status': 'authenticated'
        })

    async def broadcast(self, topic: str, message: Any, exclude: List[str] = None) -> None:
        """
        Broadcast a message to all clients subscribed to a topic.
        
        Args:
            topic: Topic to broadcast to
            message: Message to send (will be JSON-serialized)
            exclude: List of client IDs to exclude from the broadcast
        """
        if topic not in self.subscriptions:
            return
            
        exclude = set(exclude or [])
        clients = self.subscriptions[topic] - exclude
        
        if not clients:
            return
            
        try:
            message_json = json.dumps({
                'type': 'update',
                'topic': topic,
                'data': message
            })
            
            # Send to all subscribed clients
            tasks = []
            for client_id in clients:
                if client_id in self.clients:
                    tasks.append(
                        self._send_raw(client_id, message_json)
                    )
                    
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.stats['messages_sent'] += len(tasks)
                
        except Exception as e:
            logger.error(f"Error broadcasting message: {str(e)}", exc_info=True)
            self.stats['errors'] += 1

    async def _send_message(self, client_id: str, message: Any) -> None:
        """Send a JSON message to a client."""
        try:
            await self._send_raw(client_id, json.dumps(message))
            self.stats['messages_sent'] += 1
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}", exc_info=True)
            self.stats['errors'] += 1
            raise

    async def _send_raw(self, client_id: str, message: str) -> None:
        """Send a raw message to a client."""
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not found")
            return
            
        try:
            await self.clients[client_id].send(message)
        except ConnectionClosed:
            await self._cleanup_client(client_id)
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {str(e)}")
            await self._cleanup_client(client_id)
            raise

    async def _send_error(self, client_id: str, error: str, code: int = 400) -> None:
        """Send an error message to a client."""
        await self._send_message(client_id, {
            'type': 'error',
            'code': code,
            'message': error
        })

    async def _close_client(self, websocket, code: int, reason: str) -> None:
        """Close a client connection."""
        try:
            await websocket.close(code, reason)
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")

    async def _cleanup_client(self, client_id: str) -> None:
        """Clean up resources for a disconnected client."""
        if client_id in self.clients:
            del self.clients[client_id]
            self.stats['connections'] = len(self.clients)
            
        # Remove from all subscriptions
        for topic in list(self.subscriptions.keys()):
            if client_id in self.subscriptions[topic]:
                self.subscriptions[topic].remove(client_id)
                
            # Clean up empty topics
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            **self.stats,
            'clients_connected': len(self.clients),
            'topics_active': len(self.subscriptions)
        }

    # Helper methods for common message types
    async def send_order_update(self, order: Order) -> None:
        """Send an order update to all subscribed clients."""
        await self.broadcast(f'orders:{order.symbol}', asdict(order))
        
    async def send_position_update(self, position: Position) -> None:
        """Send a position update to all subscribed clients."""
        await self.broadcast(f'positions:{position.symbol}', asdict(position))
        
    async def send_account_update(self, account: AccountInfo) -> None:
        """Send an account update to all subscribed clients."""
        await self.broadcast('account', asdict(account))
        
    async def send_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Send market data to all subscribed clients."""
        await self.broadcast(f'market:{symbol}', data)

# Example usage
async def main():
    # Create and start server
    server = WebSocketServer(port=8765)
    await server.start()
    
    # Example authentication function
    def authenticate(username: str, password: str) -> bool:
        return username == "admin" and password == "password"
    
    server.authenticate = authenticate
    
    try:
        # Keep the server running
        while True:
            # Example: Broadcast server stats every 60 seconds
            await asyncio.sleep(60)
            stats = server.get_stats()
            logger.info(f"Server stats: {stats}")
            
    except asyncio.CancelledError:
        await server.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass