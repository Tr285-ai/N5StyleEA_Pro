# messaging/order_queue.py
import json
import logging
from typing import Dict, Any, Optional, List
import redis
from datetime import datetime
import uuid
import asyncio
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

@dataclass
class OrderMessage:
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    order_type: str = 'limit'  # 'limit', 'market', 'stop', etc.
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class OrderQueue:
    """Redis-based message queue for order processing."""
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379/0",
        stream_name: str = "order_stream",
        consumer_group: str = "order_processor",
        consumer_name: str = None
    ):
        """
        Initialize the order queue.
        
        Args:
            redis_url: Redis connection URL
            stream_name: Name of the Redis stream
            consumer_group: Name of the consumer group
            consumer_name: Name of this consumer instance
        """
        self.redis = redis.Redis.from_url(redis_url)
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"
        self._ensure_consumer_group()
        
    def _ensure_consumer_group(self) -> None:
        """Ensure the consumer group exists."""
        try:
            self.redis.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id='0',
                mkstream=True
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {self.consumer_group} already exists")
            else:
                raise

    async def publish_order(self, order: OrderMessage) -> str:
        """
        Publish an order to the stream.
        
        Args:
            order: OrderMessage instance
            
        Returns:
            Message ID
        """
        message = asdict(order)
        message_id = self.redis.xadd(
            name=self.stream_name,
            fields={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                   for k, v in message.items()}
        )
        logger.info(f"Published order {order.order_id} to stream {self.stream_name}")
        return message_id.decode()

    async def process_orders(
        self,
        callback: callable,
        count: int = 10,
        block: int = 1000,
        last_id: str = ">"
    ) -> None:
        """
        Process orders from the stream.
        
        Args:
            callback: Function to process messages (async function)
            count: Number of messages to process in one batch
            block: Block time in milliseconds
            last_id: Last processed message ID
        """
        while True:
            try:
                # Read messages from the stream
                messages = self.redis.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: last_id},
                    count=count,
                    block=block
                )
                
                if not messages:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Process each message
                for stream, message_list in messages:
                    for message_id, message_data in message_list:
                        try:
                            # Convert message data back to dict
                            order_data = {
                                k: json.loads(v) if isinstance(v, bytes) else v
                                for k, v in message_data.items()
                                if not k.startswith(b'_')
                            }
                            
                            # Process the order
                            await callback(order_data)
                            
                            # Acknowledge the message
                            self.redis.xack(
                                stream_name=self.stream_name,
                                groupname=self.consumer_group,
                                *[message_id]
                            )
                            
                            # Update last processed ID
                            last_id = message_id
                            
                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            
            except Exception as e:
                logger.error(f"Error in order processing loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors

    async def get_pending_orders(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get pending orders that haven't been acknowledged."""
        pending = self.redis.xpending_range(
            name=self.stream_name,
            groupname=self.consumer_group,
            min='-',
            max='+',
            count=count,
            consumername=None
        )
        return [dict(order) for order in pending]