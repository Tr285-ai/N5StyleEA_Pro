"""
Advanced risk management module with VaR, position sizing, and drawdown protection.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RiskMetric(Enum):
    """Supported risk metrics for position sizing."""
    FIXED = "fixed"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"
    VOLATILITY = "volatility"

@dataclass
class PositionSizing:
    """Handles position sizing based on different risk metrics."""
    
    @staticmethod
    def fixed_fraction(account_balance: float, risk_per_trade: float = 0.01) -> float:
        """
        Fixed fraction position sizing.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Fraction of account to risk per trade (default: 1%)
            
        Returns:
            Position size in account currency
        """
        return account_balance * risk_per_trade
    
    @staticmethod
    def kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
        """
        Kelly Criterion position sizing.
        
        Args:
            win_rate: Historical win rate (0-1)
            win_loss_ratio: Average win / average loss
            
        Returns:
            Fraction of account to risk
        """
        if win_loss_ratio <= 0:
            return 0.0
        return win_rate - ((1 - win_rate) / win_loss_ratio)
    
    @staticmethod
    def optimal_f(returns: List[float]) -> float:
        """
        Optimal f position sizing (Ralph Vince).
        
        Args:
            returns: List of historical returns
            
        Returns:
            Optimal fraction to risk
        """
        if not returns:
            return 0.0
            
        returns = np.array(returns)
        max_loss = abs(min(returns))
        
        if max_loss == 0:
            return 0.0
            
        f = np.linspace(0.01, 0.99, 100)
        growth_rates = []
        
        for fi in f:
            growth = np.prod(1.0 + (fi * returns / max_loss))
            growth_rates.append(growth)
            
        optimal_f = f[np.argmax(growth_rates)]
        return optimal_f * 0.5  # Use half-kelly for more conservative approach
    
    @staticmethod
    def volatility_scaling(volatility: float, 
                          target_vol: float = 0.2,
                          max_leverage: float = 5.0) -> float:
        """
        Position sizing based on volatility scaling.
        
        Args:
            volatility: Current market volatility (annualized)
            target_vol: Target portfolio volatility
            max_leverage: Maximum allowed leverage
            
        Returns:
            Position size multiplier
        """
        if volatility <= 0:
            return 1.0
            
        scale = min(target_vol / volatility, max_leverage)
        return max(0.1, scale)  # Minimum 10% position

class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self, 
                 max_drawdown: float = 0.2,
                 max_position_size: float = 0.1,
                 max_leverage: float = 5.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize risk manager.
        
        Args:
            max_drawdown: Maximum allowed drawdown (0-1)
            max_position_size: Maximum position size as fraction of account
            max_leverage: Maximum allowed leverage
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.peak_equity = 0
        self.drawdown = 0
        self.risk_metrics = {}
        
    def update_equity(self, equity: float) -> None:
        """Update equity and calculate drawdown."""
        self.peak_equity = max(self.peak_equity, equity)
        self.drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
    
    def calculate_var(self, 
                     returns: List[float], 
                     confidence_level: float = 0.95,
                     time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: List of historical returns
            confidence_level: Confidence level (0-1)
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR metrics
        """
        if not returns:
            return {
                'var_absolute': 0,
                'var_percentage': 0,
                'cvar_absolute': 0,
                'cvar_percentage': 0
            }
            
        returns = np.array(returns)
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        cvar = -returns[returns <= -var].mean() if (returns <= -var).any() else var
        
        # Scale for time horizon
        var_scaled = var * np.sqrt(time_horizon)
        cvar_scaled = cvar * np.sqrt(time_horizon)
        
        return {
            'var_absolute': var_scaled,
            'var_percentage': var_scaled * 100,
            'cvar_absolute': cvar_scaled,
            'cvar_percentage': cvar_scaled * 100,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon
        }
    
    def calculate_position_size(self,
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              risk_metric: RiskMetric = RiskMetric.FIXED,
                              **kwargs) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price of the asset
            stop_loss: Stop loss price
            risk_metric: Risk metric to use for position sizing
            **kwargs: Additional parameters for specific risk metrics
            
        Returns:
            Tuple of (position_size, position_value)
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0, 0.0
            
        risk_amount = account_balance * self.max_drawdown * (1 - self.drawdown)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_metric == RiskMetric.FIXED:
            risk_fraction = kwargs.get('risk_per_trade', 0.01)
            risk_amount = account_balance * risk_fraction
        
        elif risk_metric == RiskMetric.KELLY:
            win_rate = kwargs.get('win_rate', 0.5)
            win_loss_ratio = kwargs.get('win_loss_ratio', 1.0)
            kelly_fraction = PositionSizing.kelly_criterion(win_rate, win_loss_ratio)
            risk_amount = account_balance * min(kelly_fraction, 0.1)  # Cap at 10% of account
            
        elif risk_metric == RiskMetric.OPTIMAL_F:
            returns = kwargs.get('returns', [])
            if returns:
                optimal_f = PositionSizing.optimal_f(returns)
                risk_amount = account_balance * optimal_f
                
        elif risk_metric == RiskMetric.VOLATILITY:
            volatility = kwargs.get('volatility', 0.2)
            target_vol = kwargs.get('target_vol', 0.2)
            scale = PositionSizing.volatility_scaling(volatility, target_vol, self.max_leverage)
            risk_amount = account_balance * (1.0 / scale)
        
        # Calculate position size
        position_size = max(0, min(
            risk_amount / max(risk_per_share, 0.0001),  # Avoid division by zero
            (account_balance * self.max_position_size) / max(entry_price, 0.0001)
        ))
        
        position_value = position_size * entry_price
        return position_size, position_value
    
    def check_drawdown_limits(self, current_equity: float) -> bool:
        """
        Check if drawdown limits are exceeded.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            True if drawdown is within limits, False otherwise
        """
        self.update_equity(current_equity)
        return self.drawdown <= self.max_drawdown
    
    def adjust_risk_parameters(self, 
                             market_volatility: float,
                             win_rate: Optional[float] = None,
                             win_loss_ratio: Optional[float] = None) -> None:
        """
        Dynamically adjust risk parameters based on market conditions.
        
        Args:
            market_volatility: Current market volatility
            win_rate: Current strategy win rate
            win_loss_ratio: Current win/loss ratio
        """
        # Reduce position size in high volatility
        if market_volatility > 0.3:  # 30% annualized volatility
            self.max_position_size = min(0.05, self.max_position_size * 0.8)
        
        # Adjust based on strategy performance
        if win_rate is not None and win_loss_ratio is not None:
            if win_rate < 0.4 or win_loss_ratio < 1.0:
                self.max_position_size *= 0.8  # Reduce position size for poor performance
            elif win_rate > 0.6 and win_loss_ratio > 1.5:
                self.max_position_size = min(0.15, self.max_position_size * 1.1)  # Slightly increase for good performance

# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager(
        max_drawdown=0.2,
        max_position_size=0.1,
        max_leverage=5.0
    )
    
    # Example VaR calculation
    returns = np.random.normal(0.001, 0.02, 1000)  # Simulated returns
    var_results = risk_manager.calculate_var(returns)
    print(f"95% 1-day VaR: {var_results['var_percentage']:.2f}%")
    
    # Example position sizing
    account_balance = 10000
    entry_price = 100
    stop_loss = 95
    
    # Fixed fraction position sizing
    size1, value1 = risk_manager.calculate_position_size(
        account_balance, entry_price, stop_loss,
        risk_metric=RiskMetric.FIXED,
        risk_per_trade=0.02  # Risk 2% per trade
    )
    print(f"Fixed fraction position: {size1:.2f} shares (${value1:.2f})")
    
    # Kelly Criterion position sizing
    size2, value2 = risk_manager.calculate_position_size(
        account_balance, entry_price, stop_loss,
        risk_metric=RiskMetric.KELLY,
        win_rate=0.6,
        win_loss_ratio=1.5
    )
    print(f"Kelly position: {size2:.2f} shares (${value2:.2f})")
    
    # Check drawdown
    risk_manager.update_equity(9000)  # Current equity
    print(f"Current drawdown: {risk_manager.drawdown*100:.2f}%")
    print(f"Within limits: {risk_manager.check_drawdown_limits(9000)}")