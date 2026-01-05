import logging
import os
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger('trading.risk')

__path__ = [os.path.join(os.path.dirname(__file__), 'risk')]

class RiskManager:
    """Manages trading risk and position sizing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('risk_management', {})
        self.trading_config = config.get('trading', {})
        
        # Risk parameters
        self.max_risk_per_trade = self.trading_config.get('max_risk_per_trade', 0.01)  # 1% of balance
        self.max_daily_loss = self.trading_config.get('max_daily_loss', 0.05)  # 5% of balance
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% of balance
        self.stop_loss_pips = self.config.get('stop_loss_pips', 20)
        self.take_profit_pips = self.config.get('take_profit_pips', 40)
        
        # Track daily metrics
        self.daily_pnl = 0.0
        self.open_positions = []
        
        logger.info("Risk Manager initialized")

    def check_risk(self, signal: Dict[str, Any]) -> bool:
        """
        Check if a trade meets risk management criteria.
        
        Args:
            signal: Trade signal dictionary
            
        Returns:
            bool: True if trade passes risk checks, False otherwise
        """
        try:
            # Check maximum position size
            position_size = float(signal.get('amount', 0))
            if position_size > self.max_position_size:
                logger.warning(f"Position size {position_size} exceeds maximum {self.max_position_size}")
                return False
                
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
                return False
                
            # Check stop loss and take profit
            if 'stop_loss' not in signal or 'take_profit' not in signal:
                logger.warning("Missing stop loss or take profit in signal")
                return False
                
            # Additional risk checks can be added here
            # (e.g., maximum open trades, correlation with existing positions, etc.)
            
            return True
            
        except Exception as e:
            logger.error(f"Risk check failed: {str(e)}")
            return False

    def update_position(self, position: Dict[str, Any]) -> None:
        """
        Update the risk manager with a new position.
        
        Args:
            position: Dictionary with position details
        """
        self.open_positions.append(position)
        logger.info(f"Position opened: {position}")

    def update_pnl(self, pnl: float) -> None:
        """
        Update profit/loss.
        
        Args:
            pnl: Profit/loss amount (positive for profit, negative for loss)
        """
        self.daily_pnl += pnl
        logger.info(f"Updated P&L: {self.daily_pnl}")

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at the start of a new trading day)."""
        self.daily_pnl = 0.0
        self.open_positions = []
        logger.info("Daily metrics reset")