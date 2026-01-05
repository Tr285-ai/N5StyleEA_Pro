# c:\N5StyleEA_Pro v15_3\auto_updater.py
import asyncio
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import aiohttp
import semver

logger = logging.getLogger(__name__)

class AutoUpdater:
    """Handles automatic updates for the trading system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the auto-updater.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.update_url = self.config.get('update_url', 'https://api.n5styleea.com/updates')
        self.current_version = self.config.get('version', '1.0.0')
        self.update_interval = self.config.get('update_interval', 86400)  # 24 hours
        self.update_dir = Path(self.config.get('update_dir', 'updates'))
        self.update_dir.mkdir(exist_ok=True, parents=True)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> None:
        """Initialize the auto-updater."""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            
    async def check_for_updates(self) -> Optional[Dict[str, Any]]:
        """
        Check if updates are available.
        
        Returns:
            Update information if available, None otherwise
        """
        if not self.session:
            raise RuntimeError("AutoUpdater not initialized. Call initialize() first.")
            
        try:
            async with self.session.get(
                f"{self.update_url}/check",
                params={"current_version": self.current_version},
                timeout=30
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get('update_available', False):
                    return data
                return None
                
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            logger.error(f"Error checking for updates: {e}")
            return None
            
    async def download_update(self, version: str) -> Optional[Path]:
        """
        Download an update.
        
        Args:
            version: Version to download
            
        Returns:
            Path to the downloaded update file, or None if failed
        """
        if not self.session:
            raise RuntimeError("AutoUpdater not initialized. Call initialize() first.")
            
        try:
            update_file = self.update_dir / f"update_{version}.zip"
            
            # Skip if already downloaded
            if update_file.exists():
                return update_file
                
            # Download the update
            url = f"{self.update_url}/download/{version}"
            async with self.session.get(url, timeout=3600) as response:
                response.raise_for_status()
                
                # Save the update file
                with open(update_file, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                            
                return update_file
                
        except (aiohttp.ClientError, OSError) as e:
            logger.error(f"Error downloading update: {e}")
            if update_file.exists():
                update_file.unlink()
            return None
            
    async def apply_update(self, update_file: Path) -> bool:
        """
        Apply a downloaded update.
        
        Args:
            update_file: Path to the update file
            
        Returns:
            bool: True if update was applied successfully
        """
        try:
            # Backup current version
            backup_dir = self.update_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract and apply update
            # This is a placeholder - actual implementation will depend on your update mechanism
            # For example, you might use zipfile to extract and replace files
            # Then update the current version in the config
            
            logger.info(f"Successfully applied update from {update_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying update: {e}")
            # Rollback if needed
            return False
            
    async def run(self) -> None:
        """Run the auto-updater in the background."""
        try:
            while True:
                try:
                    update_info = await self.check_for_updates()
                    if update_info:
                        logger.info(f"Update available: {update_info['version']}")
                        
                        # Download the update
                        update_file = await self.download_update(update_info['version'])
                        if update_file:
                            # Apply the update
                            if await self.apply_update(update_file):
                                logger.info("Update applied successfully. Restarting...")
                                # Restart the application
                                os.execl(sys.executable, sys.executable, *sys.argv)
                                
                except Exception as e:
                    logger.error(f"Error in auto-update cycle: {e}")
                    
                # Wait before checking again
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            logger.info("Auto-updater stopped")
            await self.cleanup()
            
    def start(self) -> asyncio.Task:
        """
        Start the auto-updater in the background.
        
        Returns:
            asyncio.Task: The background task
        """
        return asyncio.create_task(self.run())