# exchanges/implementations.py
from typing import Dict, List, Optional
import ccxt
import asyncio
from decimal import Decimal
from datetime import datetime

class ExchangeWrapper:
    def __init__(self, exchange_class, api_key: str = None, api_secret: str = None, **kwargs):
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            **kwargs
        })
        self.exchange.load_markets()

    async def get_balance(self) -> Dict[str, float]:
        balance = await self.exchange.fetch_balance()
        return {k: float(v['free']) for k, v in balance['total'].items() if v['free'] > 0}

    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        ticker = await self.exchange.fetch_ticker(symbol)
        return {
            'symbol': symbol,
            'bid': float(ticker['bid']),
            'ask': float(ticker['ask']),
            'last': float(ticker['last']),
            'volume': float(ticker['baseVolume']),
            'timestamp': ticker['timestamp']
        }

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> dict:
        return await self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price
        )

class BinanceExchange(ExchangeWrapper):
    def __init__(self, api_key: str = None, api_secret: str = None, **kwargs):
        super().__init__(ccxt.binance, api_key, api_secret, {
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            },
            **kwargs
        })

class FTXExchange(ExchangeWrapper):
    def __init__(self, api_key: str = None, api_secret: str = None, **kwargs):
        super().__init__(ccxt.ftx, api_key, api_secret, {
            'enableRateLimit': True,
            'headers': {
                'FTX-SUBACCOUNT': kwargs.pop('subaccount', None)
            },
            **kwargs
        })

class KrakenExchange(ExchangeWrapper):
    def __init__(self, api_key: str = None, api_secret: str = None, **kwargs):
        super().__init__(ccxt.kraken, api_key, api_secret, {
            'options': {
                'trading_agreement': 'agree',
                'tier': 'Intermediate'
            },
            **kwargs
        })

    async def get_balance(self) -> Dict[str, float]:
        balance = await super().get_balance()
        # Kraken uses XBT instead of BTC
        if 'XBT' in balance:
            balance['BTC'] = balance.pop('XBT')
        return balance