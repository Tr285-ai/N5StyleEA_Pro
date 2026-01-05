# services/base_service.py
import asyncio
import logging
import signal
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import time
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class ServiceConfig:
    """Configuration for a microservice."""
    name: str
    version: str = "1.0.0"
    health_check_interval: int = 30
    max_restart_attempts: int = 3
    restart_delay: int = 5

class BaseService:
    """Base class for microservices."""
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize the microservice.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.status = ServiceStatus.STOPPED
        self._shutdown_event = asyncio.Event()
        self._tasks = []
        
    async def start(self) -> None:
        """Start the service."""
        if self.status != ServiceStatus.STOPPED:
            logger.warning(f"Service {self.config.name} is already {self.status.value}")
            return
            
        self.status = ServiceStatus.STARTING
        logger.info(f"Starting service {self.config.name}...")
        
        try:
            # Register signal handlers
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda: asyncio.create_task(self.stop())
                )
                
            # Initialize the service
            await self.initialize()
            
            # Start the main loop
            self._tasks = [
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._run())
            ]
            
            self.status = ServiceStatus.RUNNING
            logger.info(f"Service {self.config.name} started successfully")
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"Error in service {self.config.name}: {e}", exc_info=True)
            raise
            
        finally:
            await self.cleanup()
            self.status = ServiceStatus.STOPPED
            logger.info(f"Service {self.config.name} stopped")
    
    async def stop(self) -> None:
        """Stop the service gracefully."""
        if self.status != ServiceStatus.RUNNING:
            return
            
        self.status = ServiceStatus.STOPPING
        logger.info(f"Stopping service {self.config.name}...")
        self._shutdown_event.set()
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
    
    async def initialize(self) -> None:
        """Initialize the service (to be implemented by subclasses)."""
        pass
        
    async def _run(self) -> None:
        """Main service loop (to be implemented by subclasses)."""
        while self.status == ServiceStatus.RUNNING:
            try:
                await self.run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in service loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def run(self) -> None:
        """Run one iteration of the service loop (to be implemented by subclasses)."""
        await asyncio.sleep(1)  # Default implementation just sleeps
    
    async def _health_check_loop(self) -> None:
        """Periodically check service health."""
        while not self._shutdown_event.is_set():
            try:
                health = await self.health_check()
                if not health["healthy"]:
                    logger.warning(f"Health check failed: {health.get('message', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                
            await asyncio.sleep(self.config.health_check_interval)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the service (can be overridden by subclasses)."""
        return {
            "service": self.config.name,
            "status": self.status.value,
            "timestamp": int(time.time()),
            "healthy": self.status == ServiceStatus.RUNNING
        }
    
    async def cleanup(self) -> None:
        """Clean up resources (can be overridden by subclasses)."""
        pass