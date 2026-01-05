# core_integration.py
import asyncio
import logging
from typing import Optional, Dict, Any

# Import the components we're integrating
from auto_updater import AutoUpdater
from system_monitor import SystemMonitor
from approval import ApprovalClient, TradeSignal

class CoreIntegration:
    """
    Core integration class that ties together auto-updating, system monitoring,
    and trade approval functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the core integration.
        
        Args:
            config: Configuration dictionary with settings for all components
        """
        self.logger = self._setup_logging()
        self.config = config
        
        # Initialize components
        self.auto_updater = AutoUpdater(config.get('updater', {}))
        self.system_monitor = SystemMonitor(config.get('monitor', {}))
        self.approval_client = ApprovalClient(
            server_url=config.get('approval', {}).get('server_url', ''),
            api_key=config.get('approval', {}).get('api_key', '')
        )
        
        # State
        self.running = False
        self.trade_signals = []
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the core integration."""
        logger = logging.getLogger('CoreIntegration')
        logger.setLevel(logging.INFO)
        return logger
    
    async def start(self):
        """Start all core services."""
        self.logger.info("Starting core integration services...")
        self.running = True
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        self.logger.info("System monitoring started")
        
        # Check for updates
        if self.config.get('updater', {}).get('auto_update', True):
            await self.check_for_updates()
            
        self.logger.info("Core integration services started successfully")
        
    async def stop(self):
        """Stop all core services."""
        self.logger.info("Stopping core integration services...")
        self.running = False
        
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        self.logger.info("System monitoring stopped")
        
    async def check_for_updates(self) -> bool:
        """Check for and apply updates if available."""
        try:
            if await self.auto_updater.check_for_updates():
                if self.auto_updater.apply_updates():
                    self.logger.info("Updates applied successfully")
                    return True
        except Exception as e:
            self.logger.error(f"Error during update check: {str(e)}")
        return False
    
    async def request_trade_approval(self, trade_signal: TradeSignal) -> bool:
        """
        Request approval for a trade.
        
        Args:
            trade_signal: The trade signal to get approval for
            
        Returns:
            bool: True if approved, False otherwise
        """
        if not self.config.get('approval', {}).get('enabled', True):
            return True  # Auto-approve if approval system is disabled
            
        try:
            token = await self.approval_client.request_approval(trade_signal)
            if token:
                self.logger.info(f"Trade approval requested. Token: {token}")
                return await self.approval_client.wait_for_approval(token)
        except Exception as e:
            self.logger.error(f"Error in trade approval: {str(e)}")
            return False
        return False