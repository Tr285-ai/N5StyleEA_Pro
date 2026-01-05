# broker_client.py
import requests
import time
import hmac
import hashlib
import json
from typing import Dict, Optional, List, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("BrokerClient")

class BrokerClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str, demo_mode: bool = True):
        """
        Initialize the broker API client.
        
        Args:
            api_key: Your broker API key
            api_secret: Your broker API secret
            base_url: Base URL for the broker's API
            demo_mode: Whether to use demo account
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.demo_mode = demo_mode
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        })
        self.account_info = None
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _get_signature(self, params: dict) -> str:
        """Generate API signature for authentication"""
        param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _request(self, method: str, endpoint: str, params: Optional[dict] = None, 
                data: Optional[dict] = None) -> dict:
        """Make an API request"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}
        params.update({
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        })
        
        # Add signature
        params['signature'] = self._get_signature(params)
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, params=params, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"{error_msg} - {error_data.get('msg', '')}"
                except:
                    error_msg = f"{error_msg} - {e.response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # Account Methods
    def get_account_info(self) -> dict:
        """Get account information"""
        try:
            response = self._request('GET', '/account')
            self.account_info = response
            return response
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    # Market Data Methods
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 500) -> List[list]:
        """
        Get kline/candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, 1d, etc.)
            limit: Number of candles to return (max 1000)
            
        Returns:
            List of klines in format:
            [
                [open_time, open, high, low, close, volume, close_time, ...],
                ...
            ]
        """
        try:
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': min(1000, max(10, limit))
            }
            return self._request('GET', '/klines', params=params)
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            raise

    def get_ticker(self, symbol: str) -> dict:
        """Get 24hr ticker price change statistics"""
        try:
            params = {'symbol': symbol.upper()}
            return self._request('GET', '/ticker/24hr', params=params)
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise

    # Trading Methods
    def create_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> dict:
        """
        Create a new order
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            order_type: 'MARKET', 'LIMIT', 'STOP_LOSS', etc.
            quantity: Amount to buy/sell
            price: Price (required for LIMIT orders)
            stop_price: Stop price (for STOP_LOSS, TAKE_PROFIT, etc.)
            
        Returns:
            Order details
        """
        try:
            data = {
                'symbol': symbol.upper(),
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': self._format_quantity(symbol, quantity),
                'timestamp': int(time.time() * 1000)
            }
            
            if price is not None:
                data['price'] = self._format_price(symbol, price)
            if stop_price is not None:
                data['stopPrice'] = self._format_price(symbol, stop_price)
                
            return self._request('POST', '/order', data=data)
            
        except Exception as e:
            logger.error(f"Failed to create {side} order for {symbol}: {e}")
            raise

    def close_position(self, symbol: str, position_side: str = 'BOTH') -> dict:
        """
        Close all open positions for a symbol
        
        Args:
            symbol: Trading pair
            position_side: 'LONG', 'SHORT', or 'BOTH'
            
        Returns:
            Order details
        """
        try:
            data = {
                'symbol': symbol.upper(),
                'positionSide': position_side.upper(),
                'timestamp': int(time.time() * 1000)
            }
            return self._request('POST', '/closePosition', data=data)
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """Get all open orders (optionally filtered by symbol)"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol.upper()
            return self._request('GET', '/openOrders', params=params)
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise

    def get_position_risk(self, symbol: Optional[str] = None) -> List[dict]:
        """Get position risk (optionally filtered by symbol)"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol.upper()
            return self._request('GET', '/positionRisk', params=params)
        except Exception as e:
            logger.error(f"Failed to get position risk: {e}")
            raise

    # Helper Methods
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol's lot size rules"""
        # In a real implementation, you would get the lot size from exchange info
        # and round the quantity to the appropriate precision
        return f"{quantity:.8f}".rstrip('0').rstrip('.')

    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol's tick size rules"""
        # In a real implementation, you would get the tick size from exchange info
        # and round the price to the appropriate precision
        return f"{price:.8f}".rstrip('0').rstrip('.')

    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        try:
            account = self.get_account_info()
            balances = {}
            for asset in account.get('balances', []):
                free = float(asset.get('free', 0))
                locked = float(asset.get('locked', 0))
                if free > 0 or locked > 0:
                    balances[asset['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            return balances
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get current position for a symbol"""
        try:
            positions = self.get_position_risk(symbol)
            for pos in positions:
                if pos.get('symbol') == symbol.upper():
                    return {
                        'symbol': pos['symbol'],
                        'position_side': pos.get('positionSide'),
                        'position_amt': float(pos.get('positionAmt', 0)),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'leverage': int(pos.get('leverage', 1)),
                        'unrealized_pnl': float(pos.get('unRealizedProfit', 0))
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise

    def get_historical_trades(self, symbol: str, limit: int = 500) -> List[dict]:
        """Get historical trades"""
        try:
            params = {
                'symbol': symbol.upper(),
                'limit': min(1000, max(1, limit))
            }
            return self._request('GET', '/myTrades', params=params)
        except Exception as e:
            logger.error(f"Failed to get historical trades for {symbol}: {e}")
            raise

    def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Check order status"""
        try:
            params = {
                'symbol': symbol.upper(),
                'orderId': order_id
            }
            return self._request('GET', '/order', params=params)
        except Exception as e:
            logger.error(f"Failed to get status for order {order_id}: {e}")
            raise