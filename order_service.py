# services/order_service.py
from typing import Dict, Any, Optional
import logging
from .base_service import BaseService, ServiceConfig
from messaging.order_queue import OrderQueue, OrderMessage
from caching.redis_cache import RedisCache
import json

logger = logging.getLogger(__name__)

class OrderService(BaseService):
    """Microservice for processing orders."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize the order service."""
        config = ServiceConfig(
            name="order-service",
            version="1.0.0"
        )
        super().__init__(config)
        self.redis_url = redis_url
        self.order_queue = None
        self.cache = None
    
    async def initialize(self) -> None:
        """Initialize the service."""
        # Initialize Redis cache
        self.cache = RedisCache(redis_url=self.redis_url)
        
        # Initialize order queue
        self.order_queue = OrderQueue(
            redis_url=self.redis_url,
            stream_name="orders",
            consumer_group="order_processors"
        )
        
        logger.info("Order service initialized")
    
    async def process_order(self, order_data: Dict[str, Any]) -> None:
        """Process an order from the queue."""
        try:
            order = OrderMessage(**order_data)
            logger.info(f"Processing order {order.order_id} for {order.symbol} {order.side} {order.quantity} @ {order.price}")
            
            # Here you would implement your order processing logic
            # For example, validate the order, check risk limits, etc.
            
            # Cache the order status
            await self.cache.set(
                f"order:{order.order_id}",
                {
                    "status": "processing",
                    "symbol": order.symbol,
                    "side": order.side,
                    "price": order.price,
                    "quantity": order.quantity,
                    "timestamp": order.timestamp
                },
                ttl=86400  # 24 hours
            )
            
            # Simulate order processing
            await asyncio.sleep(0.1)
            
            # Update order status
            await self.cache.set(
                f"order:{order.order_id}",
                {
                    "status": "completed",
                    "symbol": order.symbol,
                    "side": order.side,
                    "price": order.price,
                    "quantity": order.quantity,
                    "timestamp": order.timestamp,
                    "completed_at": OrderMessage.timestamp
                },
                ttl=86400
            )
            
            logger.info(f"Order {order.order_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing order: {e}")
            # Handle error (e.g., move to dead letter queue)
    
    async def run(self) -> None:
        """Main service loop."""
        if not self.order_queue:
            raise RuntimeError("Order queue not initialized")
            
        # Process orders from the queue
        await self.order_queue.process_orders(
            callback=self.process_order,
            count=10,
            block=1000
        )
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an order."""
        if not self.cache:
            return None
        return await self.cache.get(f"order:{order_id}")