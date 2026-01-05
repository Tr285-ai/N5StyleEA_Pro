import os
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger('trading.approval')

@dataclass
class TradeSignal:
    """Data class representing a trade signal."""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    amount: float
    expiry: int     # in seconds
    price: float = 0.0
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'amount': self.amount,
            'expiry': self.expiry,
            'price': self.price,
            'timestamp': self.timestamp
        }

class ApprovalClient:
    """Handles trade approval workflow."""
    
    def __init__(self, server_url: str = '', api_key: str = ''):
        """
        Initialize the approval client.
        
        Args:
            server_url: URL of the approval server
            api_key: API key for authentication
        """
        self.server_url = server_url
        self.api_key = api_key
        self.timeout = 30  # seconds
        self.pending_approvals = {}  # token -> (timestamp, callback)
        
        logger.info(f"Approval client initialized (Server: {server_url or 'Local'})")

    def request_approval(self, signal: TradeSignal) -> bool:
        """
        Request approval for a trade.
        
        Args:
            signal: TradeSignal object with trade details
            
        Returns:
            bool: True if approved, False if rejected or error
        """
        try:
            if not self.server_url:
                # Local approval (for testing)
                logger.info("No approval server configured. Using auto-approval.")
                return True
                
            # Prepare request
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = signal.to_dict()
            
            # Send request
            import requests
            response = requests.post(
                f"{self.server_url}/api/approval/request",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get('approved', False)
            
        except Exception as e:
            logger.error(f"Approval request failed: {str(e)}")
            return False

    def check_approval_status(self, token: str) -> Dict[str, Any]:
        """
        Check the status of a pending approval.
        
        Args:
            token: Approval token received from the server
            
        Returns:
            Dict with status information
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            import requests
            response = requests.get(
                f"{self.server_url}/api/approval/status/{token}",
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to check approval status: {str(e)}")
            return {'status': 'error', 'message': str(e)}