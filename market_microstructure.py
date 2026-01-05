import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import requests
import time
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class OrderBook:
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    timestamp: float

class MarketMicrostructureAnalyzer:
    def __init__(self, data_sources: Optional[Dict] = None):
        self.data_sources = data_sources or {}
        self.order_book_cache: Dict[str, OrderBook] = {}
        
    async def get_order_book(self, symbol: str, exchange: str = "binance") -> Optional[OrderBook]:
        """Fetch order book from the specified exchange."""
        try:
            if exchange == "binance":
                return await self._get_binance_order_book(symbol)
            elif exchange == "deriv":
                return await self._get_deriv_order_book(symbol)
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
        except Exception as e:
            logger.error(f"Error fetching {exchange} order book for {symbol}: {e}")
            return None

    async def _get_binance_order_book(self, symbol: str) -> OrderBook:
        """Fetch order book from Binance."""
        url = f"https://fapi.binance.com/fapi/v1/depth"
        params = {
            "symbol": symbol.replace("/", "").upper(),
            "limit": 50  # Get top 50 price levels
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return OrderBook(
            bids=[(float(price), float(qty)) for price, qty in data['bids']],
            asks=[(float(price), float(qty)) for price, qty in data['asks']],
            timestamp=time.time()
        )

    async def _get_deriv_order_book(self, symbol: str) -> OrderBook:
        """Fetch synthetic order book from Deriv."""
        # Implementation would be similar to _get_binance_order_book
        # but using Deriv's API endpoints
        pass

    def calculate_market_sentiment(self, order_book: OrderBook) -> Dict[str, float]:
        """Calculate market sentiment based on order book depth."""
        bids = np.array(order_book.bids)
        asks = np.array(order_book.asks)
        
        if len(bids) == 0 or len(asks) == 0:
            return {"sentiment": 0.5, "bias": "neutral", "confidence": 0}
        
        # Calculate total volume at bid and ask
        total_bid_volume = np.sum(bids[:, 1])
        total_ask_volume = np.sum(asks[:, 1])
        
        # Calculate volume-weighted average prices
        vwap_bid = np.sum(bids[:, 0] * bids[:, 1]) / total_bid_volume
        vwap_ask = np.sum(asks[:, 0] * asks[:, 1]) / total_ask_volume
        
        # Calculate order book imbalance
        total_volume = total_bid_volume + total_ask_volume
        volume_imbalance = (total_bid_volume - total_ask_volume) / total_volume
        
        # Calculate price impact
        mid_price = (vwap_bid + vwap_ask) / 2
        bid_impact = (mid_price - bids[0][0]) / mid_price
        ask_impact = (asks[0][0] - mid_price) / mid_price
        price_impact = bid_impact - ask_impact
        
        # Combine factors into sentiment score (0-1 range)
        sentiment = 0.5 + (volume_imbalance * 0.3) + (price_impact * 0.2)
        
        # Clamp between 0 and 1
        sentiment = max(0, min(1, sentiment))
        
        return {
            "sentiment": sentiment,
            "bias": "bullish" if sentiment > 0.55 else "bearish" if sentiment < 0.45 else "neutral",
            "confidence": abs(sentiment - 0.5) * 2,  # 0-1 range
            "bid_volume": total_bid_volume,
            "ask_volume": total_ask_volume,
            "vwap_bid": vwap_bid,
            "vwap_ask": vwap_ask
        }

    def detect_liquidity_clusters(self, order_book: OrderBook, price_step: float = 0.001) -> Dict:
        """Detect liquidity clusters in the order book."""
        bids = np.array(order_book.bids)
        asks = np.array(order_book.asks)
        
        if len(bids) == 0 or len(asks) == 0:
            return {"support": [], "resistance": []}
        
        # Round prices to detect clusters
        bid_prices = np.round(bids[:, 0] / price_step) * price_step
        ask_prices = np.round(asks[:, 0] / price_step) * price_step
        
        # Count volume at each price level
        bid_clusters = {}
        for price, qty in zip(bid_prices, bids[:, 1]):
            bid_clusters[price] = bid_clusters.get(price, 0) + qty
            
        ask_clusters = {}
        for price, qty in zip(ask_prices, asks[:, 1]):
            ask_clusters[price] = ask_clusters.get(price, 0) + qty
        
        # Find significant clusters (top 20% by volume)
        min_bid_vol = np.percentile(list(bid_clusters.values()), 80) if bid_clusters else 0
        min_ask_vol = np.percentile(list(ask_clusters.values()), 80) if ask_clusters else 0
        
        support = [p for p, v in bid_clusters.items() if v >= min_bid_vol]
        resistance = [p for p, v in ask_clusters.items() if v >= min_ask_vol]
        
        return {
            "support": sorted(support, reverse=True)[:5],  # Top 5 support levels
            "resistance": sorted(resistance)[:5]  # Top 5 resistance levels
        }