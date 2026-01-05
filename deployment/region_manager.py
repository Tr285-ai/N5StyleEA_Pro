"""
Multi-Region Deployment Manager

This module handles deployment and synchronization across multiple geographic regions
for low-latency global trading.
"""
import time
import json
import socket
import threading
import queue
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import requests
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Region:
    """Represents a geographic region for deployment."""
    id: str
    name: str
    endpoint: str
    location: Tuple[float, float]  # (lat, lon)
    is_active: bool = True
    last_ping: float = 0.0
    latency_ms: float = 0.0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'endpoint': self.endpoint,
            'location': self.location,
            'is_active': self.is_active,
            'latency_ms': self.latency_ms,
            'error_count': self.error_count,
            'metadata': self.metadata
        }

class RegionManager:
    """Manages multi-region deployment and routing."""
    
    def __init__(self, local_region_id: str, regions: List[Dict[str, Any]]):
        """
        Initialize the region manager.
        
        Args:
            local_region_id: ID of the local region
            regions: List of region configurations
        """
        self.local_region_id = local_region_id
        self.regions: Dict[str, Region] = {}
        self.local_region: Optional[Region] = None
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.message_handlers = {}
        
        # Initialize regions
        for region_data in regions:
            region = Region(**region_data)
            self.regions[region.id] = region
            
            if region.id == local_region_id:
                self.local_region = region
        
        if not self.local_region:
            raise ValueError(f"Local region {local_region_id} not found in regions")
        
        # Register default handlers
        self.register_handler('ping', self._handle_ping)
        self.register_handler('sync', self._handle_sync)
    
    def start(self):
        """Start the region manager."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(f"Region manager started in {self.local_region.name}")
    
    def stop(self):
        """Stop the region manager."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.executor.shutdown(wait=True)
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    def _run(self):
        """Main processing loop."""
        while self.running:
            try:
                # Process incoming messages
                self._process_messages()
                
                # Monitor region health
                self._monitor_regions()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in region manager: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def _process_messages(self):
        """Process messages from the queue."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
    
    def _handle_message(self, message: Dict[str, Any]):
        """Handle an incoming message."""
        try:
            msg_type = message.get('type')
            if msg_type in self.message_handlers:
                self.message_handlers[msg_type](message)
            else:
                logger.warning(f"No handler for message type: {msg_type}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_ping(self, message: Dict[str, Any]):
        """Handle ping message."""
        region_id = message.get('from')
        if region_id in self.regions:
            region = self.regions[region_id]
            region.last_ping = time.time()
            region.latency_ms = message.get('latency_ms', 0)
            region.is_active = True
            region.error_count = 0
    
    def _handle_sync(self, message: Dict[str, Any]):
        """Handle sync message."""
        # Implement state synchronization logic here
        pass
    
    def _monitor_regions(self):
        """Monitor health of all regions."""
        current_time = time.time()
        
        for region in self.regions.values():
            if region.id == self.local_region_id:
                continue  # Skip self
                
            # Check if region is responsive
            if current_time - region.last_ping > 30:  # 30 seconds timeout
                region.error_count += 1
                if region.error_count > 3:  # Mark as inactive after 3 failures
                    region.is_active = False
                    logger.warning(f"Region {region.name} is not responding")
    
    def send_message(self, region_id: str, message: Dict[str, Any]):
        """Send a message to a specific region."""
        if region_id not in self.regions:
            logger.error(f"Unknown region: {region_id}")
            return False
            
        region = self.regions[region_id]
        if not region.is_active:
            logger.warning(f"Cannot send message to inactive region: {region.name}")
            return False
        
        try:
            # Add metadata to message
            message.update({
                'from': self.local_region_id,
                'timestamp': datetime.utcnow().isoformat(),
                'message_id': f"{self.local_region_id}_{int(time.time() * 1000)}"
            })
            
            # Send message asynchronously
            self.executor.submit(self._send_http_request, region.endpoint, message)
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to {region.name}: {e}")
            region.error_count += 1
            if region.error_count > 3:
                region.is_active = False
            return False
    
    def broadcast(self, message: Dict[str, Any], exclude_local: bool = False):
        """Broadcast a message to all active regions."""
        results = {}
        for region_id in self.regions:
            if exclude_local and region_id == self.local_region_id:
                continue
            results[region_id] = self.send_message(region_id, message)
        return results
    
    def get_best_region(self, target_location: Tuple[float, float]) -> Optional[Region]:
        """
        Find the best region to handle a request based on geographic proximity
        and current load/latency.
        """
        best_region = None
        best_score = float('inf')
        
        for region in self.regions.values():
            if not region.is_active:
                continue
                
            # Calculate distance score (simplified)
            lat1, lon1 = target_location
            lat2, lon2 = region.location
            distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5
            
            # Consider latency and error rate
            score = distance * (1 + region.latency_ms / 1000) * (1 + region.error_count * 0.1)
            
            if score < best_score:
                best_score = score
                best_region = region
        
        return best_region
    
    def sync_state(self, state: Dict[str, Any]):
        """Synchronize state across all regions."""
        message = {
            'type': 'sync',
            'state': state,
            'version': int(time.time())
        }
        return self.broadcast(message, exclude_local=True)
    
    def _send_http_request(self, url: str, data: Dict[str, Any]) -> bool:
        """Send an HTTP request to a region endpoint."""
        try:
            response = requests.post(
                url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of all regions."""
        return {
            'local_region': self.local_region.id,
            'regions': {k: v.to_dict() for k, v in self.regions.items()},
            'timestamp': datetime.utcnow().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Example configuration
    regions_config = [
        {
            'id': 'us-east-1',
            'name': 'N. Virginia',
            'endpoint': 'http://us-east-1.example.com/api',
            'location': (39.0438, -77.4874)
        },
        {
            'id': 'eu-west-1',
            'name': 'Ireland',
            'endpoint': 'http://eu-west-1.example.com/api',
            'location': (53.3498, -6.2603)
        },
        {
            'id': 'ap-southeast-1',
            'name': 'Singapore',
            'endpoint': 'http://ap-southeast-1.example.com/api',
            'location': (1.3521, 103.8198)
        }
    ]
    
    # Initialize region manager
    region_manager = RegionManager(
        local_region_id='us-east-1',
        regions=regions_config
    )
    
    try:
        # Start the region manager
        region_manager.start()
        
        # Example: Send a message to a specific region
        region_manager.send_message('eu-west-1', {
            'type': 'test',
            'message': 'Hello from N. Virginia!'
        })
        
        # Example: Broadcast a state update
        region_manager.sync_state({
            'order_books': {},
            'positions': {},
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        region_manager.stop()
