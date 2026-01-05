# money_manager.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json

logger = logging.getLogger('MoneyManager')

@dataclass
class Position:
    symbol: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

class MoneyManager:
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.2,
        max_position_size: float = 0.1,
        max_leverage: float = 10.0
    ):
        """
        Initialize the MoneyManager with risk parameters.
        
        Args:
            initial_balance: Starting account balance
            risk_per_trade: Percentage of capital to risk per trade (0.01 = 1%)
            max_drawdown: Maximum allowed drawdown before reducing position sizes
            max_position_size: Maximum position size as fraction of account
            max_leverage: Maximum allowed leverage
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.drawdown = 0.0
        self.max_equity = initial_balance
        self.logger = logging.getLogger('MoneyManager')
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_multiplier: Adjust risk (1.0 = normal, 0.5 = half risk, etc.)
            
        Returns:
            Tuple of (position_size, position_value)
        """
        if entry_price <= 0 or stop_loss <= 0:
            raise ValueError("Entry price and stop loss must be positive")
            
        # Calculate risk amount
        risk_amount = self.balance * self.risk_per_trade * risk_multiplier
        
        # Adjust for drawdown
        drawdown_factor = 1.0 - min(self.drawdown / self.max_drawdown, 1.0)
        risk_amount *= drawdown_factor
        
        # Calculate position size
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0, 0.0
            
        position_size = risk_amount / risk_per_share
        
        # Apply position size limits
        max_size = (self.balance * self.max_position_size) / entry_price
        position_size = min(position_size, max_size)
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Check leverage
        leverage = position_value / self.balance
        if leverage > self.max_leverage:
            position_size = (self.balance * self.max_leverage) / entry_price
            position_value = position_size * entry_price
            
        return position_size, position_value
        
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_multiplier: float = 1.0
    ) -> Optional[Position]:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            risk_multiplier: Adjust risk (1.0 = normal, 0.5 = half risk, etc.)
            
        Returns:
            Position object if successful, None otherwise
        """
        if symbol in self.positions:
            self.logger.warning(f"Position for {symbol} already exists")
            return None
            
        # Calculate position size
        position_size, _ = self.calculate_position_size(
            entry_price, stop_loss, risk_multiplier
        )
        
        if position_size <= 0:
            self.logger.warning("Position size is zero or negative")
            return None
            
        # Create and store position
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.utcnow()
        )
        
        self.positions[symbol] = position
        self.logger.info(
            f"Opened {symbol} position: {position_size:.4f} @ {entry_price:.4f}, "
            f"SL: {stop_loss:.4f}, TP: {take_profit:.4f}"
        )
        
        return position
        
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Optional[float]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing (e.g., 'stop_loss', 'take_profit', 'manual')
            
        Returns:
            P&L amount if successful, None otherwise
        """
        if symbol not in self.positions:
            self.logger.warning(f"No open position for {symbol}")
            return None
            
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.size == 0:
            pnl = 0.0
            pnl_pct = 0.0
        else:
            pnl = (exit_price - position.entry_price) * position.size
            pnl_pct = (exit_price / position.entry_price - 1) * 100
            if position.size < 0:  # Short position
                pnl = -pnl
                pnl_pct = -pnl_pct
                
        # Update position
        position.exit_price = exit_price
        position.exit_time = datetime.utc()
        position.pnl = pnl
        position.pnl_pct = pnl_pct
        
        # Update account balance
        self.balance += pnl
        self.equity = self.balance  # Simplified - in reality, you'd have open positions P&L
        
        # Update max equity and drawdown
        self.max_equity = max(self.max_equity, self.equity)
        self.drawdown = (self.max_equity - self.equity) / self.max_equity
        
        # Log the trade
        trade = {
            'symbol': symbol,
            'entry_time': position.entry_time,
            'exit_time': position.exit_time,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        self.trade_history.append(trade)
        
        # Remove from active positions
        del self.positions[symbol]
        
        self.logger.info(
            f"Closed {symbol} position: P&L = {pnl:.2f} ({pnl_pct:.2f}%), Reason: {reason}"
        )
        
        return pnl
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get an open position by symbol."""
        return self.positions.get(symbol)
        
    def get_open_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return self.positions.copy()
        
    def get_account_summary(self) -> Dict[str, float]:
        """Get account summary."""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'margin_used': sum(
                p.size * p.entry_price for p in self.positions.values()
            ),
            'free_margin': self.equity - sum(
                p.size * p.entry_price for p in self.positions.values()
            ),
            'margin_level': self.equity / sum(
                p.size * p.entry_price for p in self.positions.values()
            ) if self.positions else float('inf'),
            'drawdown': self.drawdown,
            'max_drawdown': self.max_drawdown
        }
        
    def save_trade_history(self, filename: str = "trade_history.json"):
        """Save trade history to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.trade_history, f, default=str, indent=2)
            
    def load_trade_history(self, filename: str = "trade_history.json"):
        """Load trade history from a JSON file."""
        try:
            with open(filename, 'r') as f:
                self.trade_history = json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Trade history file {filename} not found")
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from trade history."""
        if not self.trade_history:
            return {}
            
        df = pd.DataFrame(self.trade_history)
        df['is_win'] = df['pnl'] > 0
        
        metrics = {
            'total_trades': len(df),
            'win_rate': df['is_win'].mean() * 100,
            'total_pnl': df['pnl'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'avg_win': df[df['is_win']]['pnl'].mean(),
            'avg_loss': df[~df['is_win']]['pnl'].mean(),
            'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum() / 
                               df[df['pnl'] < 0]['pnl'].sum()) if df[df['pnl'] < 0]['pnl'].sum() != 0 else float('inf'),
            'max_drawdown': self.drawdown * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(df),
            'sortino_ratio': self._calculate_sortino_ratio(df)
        }
        
        return metrics
        
    def _calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from trade P&Ls."""
        if len(df) < 2:
            return 0.0
            
        returns = df['pnl'] / self.initial_balance
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        return (excess_returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
        
    def _calculate_sortino_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio from trade P&Ls."""
        if len(df) < 2:
            return 0.0
            
        returns = df['pnl'] / self.initial_balance
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        return (excess_returns.mean() / (downside_std + 1e-9)) * np.sqrt(252) if downside_std > 0 else 0.0