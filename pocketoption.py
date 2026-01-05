# pocketoption.py
"""
PocketOption Broker Integration

This module implements the PocketOption broker interface based on the BaseBroker class.
It handles authentication, order management, and market data retrieval.

Author: N5StyleEA Team
Version: 15.2.1
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import aiohttp
import pandas as pd
from datetime import datetime, timedelta

from .base_broker import (
    BaseBroker, Order, OrderStatus, Position, AccountInfo, 
    OrderType, OrderSide, TimeInForce, PositionSide
)

logger = logging.getLogger(__name__)

class PocketOptionBroker(BaseBroker):
    """PocketOption broker implementation."""
    
    BASE_URL = "https://pocketoption.com/api"
    WS_URL = "wss://pocketoption.com/socket.io/?EIO=3&transport=websocket"
    
    def __init__(
        self,
        email: str,
        password: str,
        is_demo: bool = True,
        **kwargs
    ):
        """
        Initialize the PocketOption broker client.
        
        Args:
            email: PocketOption account email
            password: PocketOption account password
            is_demo: Whether to use demo account
            **kwargs: Additional arguments passed to BaseBroker
        """
        super().__init__(is_testnet=is_demo, **kwargs)
        self.email = email
        self.password = password
        self.session = None
        self.ws = None
        self.assets = {}
        self.candles_cache = {}
        self._last_request_time = 0
        self._request_delay = 0.1  # 100ms between requests to avoid rate limiting

    # Connection Management
    async def connect(self) -> bool:
        """Connect to PocketOption API."""
        try:
            if self.connected:
                return True
                
            self.session = aiohttp.ClientSession()
            
            # Login to PocketOption
            login_data = {
                'email': self.email,
                'password': self.password
            }
            
            async with self.session.post(
                f"{self.BASE_URL}/v2/login",
                json=login_data
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise ConnectionError(f"Login failed: {error}")
                
                data = await response.json()
                if not data.get('isSuccessful', False):
                    raise ConnectionError(f"Login failed: {data.get('message', 'Unknown error')}")
                
                self.session.headers.update({
                    'Authorization': f"Bearer {data['data']['ssid']}",
                    'x-app-version': '0.1.0',
                    'x-client-type': 'web',
                    'x-client-version': '0.1.0'
                })
            
            # Load assets list
            await self._load_assets()
            
            # Connect to WebSocket
            await self._connect_websocket()
            
            self.connected = True
            self._trigger_callback('on_connect')
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            await self.disconnect()
            raise

    async def disconnect(self) -> bool:
        """Disconnect from PocketOption API."""
        try:
            if self.ws:
                await self.ws.close()
                self.ws = None
                
            if self.session:
                await self.session.close()
                self.session = None
                
            self.connected = False
            self._trigger_callback('on_disconnect')
            return True
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")
            return False

    async def _connect_websocket(self):
        """Connect to PocketOption WebSocket."""
        self.ws = await self.session.ws_connect(self.WS_URL)
        asyncio.create_task(self._listen_websocket())

    async def _listen_websocket(self):
        """Listen for WebSocket messages."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_ws_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket connection closed")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            await self.reconnect()

    async def _handle_ws_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            if message.startswith('0'):
                # Skip initial handshake
                return
                
            # Parse message (skip first character which is message type)
            data = json.loads(message[1:])
            
            # Handle different message types
            if isinstance(data, list) and data[0] == 'candles':
                await self._handle_candles_update(data[1])
            elif isinstance(data, dict) and data.get('name') == 'tick':
                self._handle_tick_update(data['msg'])
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")

    # Account Methods
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        try:
            data = await self._make_request('GET', '/profile')
            profile = data['profile']
            
            return AccountInfo(
                equity=float(profile['balance']),
                balance=float(profile['balance']),
                margin_available=float(profile['balance']),
                margin_used=0.0,
                margin_level=0.0,
                leverage=1.0,
                positions=await self.get_positions()
            )
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            raise

    # Order Methods
    async def place_order(self, order: Order) -> OrderStatus:
        """Place a new order."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Prepare order parameters
            params = {
                'amount': order.quantity,
                'asset': order.symbol,
                'type': 'call' if order.side == OrderSide.BUY else 'put',
                'time': 60,  # Default to 1 minute
                'timeout': 30000  # 30 seconds timeout
            }
            
            # Add optional parameters
            if order.order_type == OrderType.LIMIT:
                params['price'] = order.price
                params['type'] = 'limit'
            elif order.order_type == OrderType.STOP:
                params['type'] = 'stop'
                params['stop_price'] = order.stop_price
            
            # Send order
            data = await self._make_request('POST', '/order', json=params)
            
            # Create order status
            return OrderStatus(
                order_id=str(data['order_id']),
                status='new',
                filled_quantity=0.0,
                remaining_quantity=order.quantity,
                timestamp=int(time.time() * 1000),
                client_order_id=order.client_order_id
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return OrderStatus(
                order_id='',
                status='rejected',
                error=str(e),
                timestamp=int(time.time() * 1000),
                client_order_id=getattr(order, 'client_order_id', '')
            )

    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an existing order."""
        try:
            await self._rate_limit()
            await self._make_request('POST', '/cancel_order', json={'order_id': order_id})
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False

    async def get_order_status(self, order_id: str, symbol: str = None) -> OrderStatus:
        """Get the status of an order."""
        try:
            await self._rate_limit()
            data = await self._make_request('GET', f'/order/{order_id}')
            
            return OrderStatus(
                order_id=order_id,
                status=data['status'],
                filled_quantity=float(data.get('filled_quantity', 0)),
                remaining_quantity=float(data.get('remaining_quantity', 0)),
                avg_fill_price=float(data.get('avg_fill_price', 0)),
                timestamp=int(time.time() * 1000)
            )
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return OrderStatus(
                order_id=order_id,
                status='error',
                error=str(e),
                timestamp=int(time.time() * 1000)
            )

    # Position Methods
    async def get_positions(self, symbol: str = None) -> List[Position]:
        """Get open positions."""
        try:
            await self._rate_limit()
            data = await self._make_request('GET', '/positions')
            
            positions = []
            for pos in data['positions']:
                if symbol and pos['asset'] != symbol:
                    continue
                    
                positions.append(Position(
                    symbol=pos['asset'],
                    quantity=float(pos['amount']),
                    avg_entry_price=float(pos['open_price']),
                    current_price=float(pos['current_price']),
                    unrealized_pnl=float(pos.get('profit', 0)),
                    side=PositionSide.LONG if pos['type'] == 'call' else PositionSide.SHORT,
                    position_id=str(pos['id']),
                    timestamp=pos.get('time', int(time.time() * 1000))
                ))
                
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise

    async def close_position(self, symbol: str, position_id: str = None) -> bool:
        """Close an open position."""
        try:
            await self._rate_limit()
            
            if position_id:
                # Close specific position
                await self._make_request('POST', '/close_position', json={'position_id': position_id})
            else:
                # Close all positions for symbol
                positions = await self.get_positions(symbol)
                for pos in positions:
                    await self._make_request('POST', '/close_position', json={'position_id': pos.position_id})
            
            return True
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False

    # Market Data Methods
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get historical price data."""
        try:
            await self._rate_limit()
            
            # Convert timeframe to minutes
            timeframe_min = self._parse_timeframe(timeframe)
            if not timeframe_min:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
                
            # Set default time range if not provided
            end = end_time or int(time.time() * 1000)
            start = start_time or (end - (timeframe_min * 60 * 1000 * limit))
            
            params = {
                'asset': symbol,
                'period': timeframe_min * 60,  # Convert to seconds
                'from': start // 1000,  # Convert to seconds
                'to': end // 1000,      # Convert to seconds
                'resolution': timeframe_min
            }
            
            data = await self._make_request('GET', '/chart', params=params)
            
            # Convert to standard format
            candles = []
            for i in range(len(data['t'])):
                candles.append({
                    'timestamp': data['t'][i] * 1000,  # Convert to milliseconds
                    'open': data['o'][i],
                    'high': data['h'][i],
                    'low': data['l'][i],
                    'close': data['c'][i],
                    'volume': data['v'][i] if 'v' in data else 0
                })
                
            return candles
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            raise

    async def get_current_price(self, symbol: str) -> float:
        """Get the current market price for a symbol."""
        try:
            await self._rate_limit()
            data = await self._make_request('GET', f'/assets/{symbol}/price')
            return float(data['price'])
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            raise

    # Utility Methods
    async def _load_assets(self):
        """Load available trading assets."""
        data = await self._make_request('GET', '/assets')
        self.assets = {asset['name']: asset for asset in data['assets']}

    def _parse_timeframe(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe = timeframe.lower()
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        return 0

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._request_delay:
            await asyncio.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an HTTP request to the PocketOption API."""
        try:
            url = f"{self.BASE_URL}{endpoint}"
            
            async with self.session.request(method, url, **kwargs) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API error ({response.status}): {error}")
                
                data = await response.json()
                if not data.get('isSuccessful', True):
                    raise Exception(f"API error: {data.get('message', 'Unknown error')}")
                
                return data.get('data', {})
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    async def _handle_candles_update(self, data: dict):
        """Handle WebSocket candles update."""
        symbol = data['asset']
        candles = data['candles']
        
        # Update cache
        if symbol not in self.candles_cache:
            self.candles_cache[symbol] = []
            
        self.candles_cache[symbol].extend(candles)
        
        # Keep only the most recent candles
        max_candles = 1000
        self.candles_cache[symbol] = self.candles_cache[symbol][-max_candles:]
        
        # Trigger callback
        self._trigger_callback('on_candle', symbol, candles)

    def _handle_tick_update(self, data: dict):
        """Handle WebSocket tick update."""
        symbol = data['asset']
        price = float(data['price'])
        timestamp = int(time.time() * 1000)
        
        # Update cache
        if symbol not in self.candles_cache:
            self.candles_cache[symbol] = []
            
        # Create a candle from the tick
        candle = {
            'timestamp': timestamp,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0
        }
        
        # Trigger callback
        self._trigger_callback('on_tick', symbol, price, timestamp)