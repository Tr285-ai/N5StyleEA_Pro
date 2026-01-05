import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

class RiskMetrics:
    def __init__(self, returns: pd.Series, confidence_level=0.95, risk_free_rate=0.0):
        self.returns = returns
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        
    def calculate_var_historical(self) -> float:
        """Calculate Value at Risk using historical simulation"""
        return -np.percentile(self.returns, 100 * (1 - self.confidence_level))
        
    def calculate_cvar(self) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var_historical()
        return -self.returns[self.returns <= -var].mean()
        
    def calculate_drawdown(self) -> pd.Series:
        """Calculate drawdown series"""
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown
        
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        return self.calculate_drawdown().min()
        
    def sharpe_ratio(self, periods=252) -> float:
        """Calculate annualized Sharpe ratio"""
        excess_returns = self.returns - self.risk_free_rate/periods
        return np.sqrt(periods) * (excess_returns.mean() / (excess_returns.std() + 1e-9))
        
    def sortino_ratio(self, periods=252) -> float:
        """Calculate annualized Sortino ratio"""
        excess_returns = self.returns - self.risk_free_rate/periods
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt((downside_returns**2).mean())
        return np.sqrt(periods) * (excess_returns.mean() / (downside_std + 1e-9))
        
    def calmar_ratio(self, years=3) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)"""
        cagr = self.calculate_cagr(years)
        max_dd = abs(self.max_drawdown())
        return cagr / (max_dd + 1e-9)
        
    def calculate_cagr(self, years=1) -> float:
        """Calculate Compound Annual Growth Rate"""
        cum_return = (1 + self.returns).prod() - 1
        return (1 + cum_return) ** (1/years) - 1
        
    def ulcer_index(self) -> float:
        """Calculate Ulcer Index"""
        dd = self.calculate_drawdown()
        return np.sqrt((dd**2).mean())
        
    def k_ratio(self) -> float:
        """Calculate K-Ratio (slope of equity curve)"""
        equity = (1 + self.returns).cumprod()
        x = np.arange(len(equity))
        slope, _ = np.polyfit(x, equity, 1)
        return slope * (252**0.5)  # Annualized
        
    def tail_ratio(self) -> float:
        """Calculate Tail Ratio (95th percentile / 5th percentile)"""
        return abs(np.percentile(self.returns, 95) / np.percentile(self.returns, 5))
        
    def plot_equity_curve(self, title="Equity Curve"):
        """Plot equity curve with drawdowns"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        equity = (1 + self.returns).cumprod()
        equity.plot(ax=ax1, title=title)
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Drawdown
        drawdown = self.calculate_drawdown()
        drawdown.plot(ax=ax2, color='red')
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig