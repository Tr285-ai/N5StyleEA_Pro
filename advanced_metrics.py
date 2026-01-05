import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float

class AdvancedMetrics:
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate / 252  # Annual to daily
    
    def calculate_all_metrics(self, trades: List[Trade]) -> Dict:
        """Calculate all performance metrics for the trading strategy."""
        if not trades:
            return {}
            
        returns = np.array([t.pnl_pct for t in trades])
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        
        return {
            'total_trades': len(trades),
            'win_rate': self._calculate_win_rate(returns),
            'profit_factor': self._calculate_profit_factor(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'average_win': np.mean(wins) if len(wins) > 0 else 0,
            'average_loss': np.mean(losses) if len(losses) > 0 else 0,
            'profit_loss_ratio': self._calculate_profit_loss_ratio(wins, losses),
            'expectancy': self._calculate_expectancy(returns),
            'kelly_criterion': self._calculate_kelly_criterion(returns),
            'value_at_risk': self._calculate_var(returns),
            'conditional_var': self._calculate_conditional_var(returns)
        }
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate the win rate of the strategy."""
        return np.mean(returns > 0) * 100
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate the profit factor (gross profits / gross losses)."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate the annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.risk_free_rate
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate the annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.risk_free_rate
        downside_returns = np.minimum(0, excess_returns)
        downside_deviation = np.std(downside_returns)
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / (downside_deviation + 1e-10)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate the maximum drawdown."""
        cum_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / (1 + peak)
        return np.min(drawdown) * 100  # Return as percentage
    
    def _calculate_profit_loss_ratio(self, wins: np.ndarray, losses: np.ndarray) -> float:
        """Calculate the average win to average loss ratio."""
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        return avg_win / avg_loss if avg_loss != 0 else float('inf')
    
    def _calculate_expectancy(self, returns: np.ndarray) -> float:
        """Calculate the expectancy of the strategy."""
        win_rate = self._calculate_win_rate(returns) / 100
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = np.mean(returns[returns <= 0]) if np.any(returns <= 0) else 0
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    def _calculate_kelly_criterion(self, returns: np.ndarray) -> float:
        """Calculate the Kelly Criterion for position sizing."""
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        return max(0.0, min(kelly, 1.0))  # Bound between 0 and 1
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) at the given confidence level."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_conditional_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR) at the given confidence level."""
        if len(returns) == 0:
            return 0.0
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])

def create_trade_analysis_report(trades: List[Trade], initial_capital: float = 10000.0) -> str:
    """Generate a comprehensive trade analysis report."""
    if not trades:
        return "No trades to analyze."
    
    metrics = AdvancedMetrics().calculate_all_metrics(trades)
    
    # Calculate equity curve
    equity = initial_capital
    equity_curve = [equity]
    for trade in trades:
        equity += trade.pnl
        equity_curve.append(equity)
    
    # Generate report
    report = f"""
    =====================
    TRADE ANALYSIS REPORT
    =====================
    
    Summary:
    ---------
    Initial Capital: ${initial_capital:,.2f}
    Final Equity: ${equity:,.2f}
    Net Profit: ${equity - initial_capital:,.2f} ({(equity/initial_capital - 1)*100:.2f}%)
    
    Performance Metrics:
    --------------------
    Total Trades: {metrics['total_trades']}
    Win Rate: {metrics['win_rate']:.2f}%
    Profit Factor: {metrics['profit_factor']:.2f}
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Sortino Ratio: {metrics['sortino_ratio']:.2f}
    Max Drawdown: {metrics['max_drawdown']:.2f}%
    Average Win: {metrics['average_win']*100:.2f}%
    Average Loss: {metrics['average_loss']*100:.2f}%
    Win/Loss Ratio: {metrics['profit_loss_ratio']:.2f}
    Expectancy: {metrics['expectancy']*100:.2f}%
    Kelly Criterion: {metrics['kelly_criterion']*100:.2f}%
    Value at Risk (95%): {metrics['value_at_risk']*100:.2f}%
    Conditional VaR (95%): {metrics['conditional_var']*100:.2f}%
    
    Trade Distribution:
    -------------------
    """
    
    # Add trade distribution by month
    trades_by_month = {}
    for trade in trades:
        month = trade.entry_time.strftime('%Y-%m')
        trades_by_month[month] = trades_by_month.get(month, 0) + 1
    
    report += "\nTrades by Month:\n"
    for month, count in sorted(trades_by_month.items()):
        report += f"  {month}: {count} trades\n"
    
    return report