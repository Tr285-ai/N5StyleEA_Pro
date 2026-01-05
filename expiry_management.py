from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class ExpiryManager:
    """Manages option expiry selection and rollover"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_expiry = None
        self.next_expiry = None
        self.expiry_dates = []
        
    async def initialize(self):
        """Initialize expiry dates and select current expiry"""
        await self._load_expiry_dates()
        self._select_current_expiry()
        
    async def _load_expiry_dates(self):
        """Load expiry dates from data source or configuration"""
        # Implementation depends on your data source
        pass
        
    def _select_current_expiry(self) -> None:
        """Select current expiry based on current date"""
        now = datetime.utcnow()
        for expiry in sorted(self.expiry_dates):
            if expiry > now:
                self.current_expiry = expiry
                self.next_expiry = self._get_next_expiry(expiry)
                break
                
    def _get_next_expiry(self, current_expiry: datetime) -> Optional[datetime]:
        """Get the next expiry date after the current one"""
        sorted_dates = sorted(self.expiry_dates)
        try:
            idx = sorted_dates.index(current_expiry)
            return sorted_dates[idx + 1] if idx + 1 < len(sorted_dates) else None
        except ValueError:
            return None
            
    def should_rollover(self, current_time: datetime) -> bool:
        """Check if it's time to roll over to the next expiry"""
        if not self.current_expiry or not self.next_expiry:
            return False
            
        # Check if we're within the rollover period (e.g., 1 day before expiry)
        rollover_time = self.current_expiry - timedelta(days=1)
        return current_time >= rollover_time
        
    async def get_expiry_chain(self, symbol: str) -> List[datetime]:
        """Get all available expiry dates for a symbol"""
        # Implementation depends on your data source
        pass