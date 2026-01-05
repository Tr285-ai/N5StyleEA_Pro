import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate the drawdown series from an equity curve.
    
    Args:
        equity_curve: Series of equity values with datetime index
        
    Returns:
        Series of drawdown values as percentages
    """
    if not isinstance(equity_curve, pd.Series):
        raise TypeError("equity_curve must be a pandas Series")
    
    if len(equity_curve) == 0:
        return pd.Series(dtype=float)
        
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess_returns.mean() / (returns.std() + 1e-10)

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
        
    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('inf')
        
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

def calculate_calmar_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        equity_curve: Series of equity values
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    if len(equity_curve) < 2:
        return 0.0
        
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
        
    annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (periods_per_year / len(returns)) - 1
    max_drawdown = calculate_drawdown(equity_curve).min()
    
    if max_drawdown == 0:
        return float('inf')
        
    return annual_return / abs(max_drawdown)

def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate the win rate from a trades DataFrame.
    
    Args:
        trades: DataFrame with 'pnl' column
        
    Returns:
        Win rate as a decimal (0-1)
    """
    if len(trades) == 0 or 'pnl' not in trades.columns:
        return 0.0
        
    winning_trades = (trades['pnl'] > 0).sum()
    return winning_trades / len(trades)

def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate the profit factor.
    
    Args:
        trades: DataFrame with 'pnl' column
        
    Returns:
        Profit factor (gross profits / gross losses)
    """
    if len(trades) == 0 or 'pnl' not in trades.columns:
        return 0.0
        
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return float('inf')
        
    return gross_profit / gross_loss

def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot the equity curve.
    
    Args:
        equity_curve: Series of equity values with datetime index
        benchmark: Optional benchmark series to plot alongside
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Plot equity curve
    plt.plot(equity_curve.index, equity_curve, label='Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark is not None:
        # Normalize benchmark to start at the same value as equity_curve
        benchmark_norm = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
        plt.plot(benchmark_norm.index, benchmark_norm, label='Benchmark', alpha=0.7)
    
    # Format plot
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format y-axis as currency
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"${x:,.2f}")
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot the drawdown curve.
    
    Args:
        equity_curve: Series of equity values with datetime index
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    drawdown = calculate_drawdown(equity_curve)
    
    plt.figure(figsize=figsize)
    plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot the distribution of returns.
    
    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    sns.histplot(returns * 100, kde=True, bins=50)
    plt.axvline(returns.mean() * 100, color='r', linestyle='--', label=f'Mean: {returns.mean()*100:.2f}%')
    plt.title(title, fontsize=14)
    plt.xlabel('Daily Return (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_report(
    metrics: Dict[str, float],
    output_dir: Union[str, Path],
    prefix: str = "backtest"
) -> None:
    """
    Generate a backtest report with metrics and visualizations.
    
    Args:
        metrics: Dictionary of performance metrics
        output_dir: Directory to save the report
        prefix: Prefix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to JSON
    metrics_file = output_dir / f"{prefix}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate markdown report
    report_file = output_dir / f"{prefix}_report.md"
    with open(report_file, 'w') as f:
        f.write(f"# Backtest Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary metrics
        f.write("## Summary Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Initial Balance | ${metrics.get('initial_balance', 0):,.2f} |\n")
        f.write(f"| Final Balance | ${metrics.get('final_balance', 0):,.2f} |\n")
        f.write(f"| Total Return | {metrics.get('total_return', 0):.2f}% |\n")
        f.write(f"| Annualized Return | {metrics.get('annualized_return', 0):.2f}% |\n")
        f.write(f"| Max Drawdown | {metrics.get('max_drawdown_pct', 0):.2f}% |\n")
        f.write(f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |\n")
        f.write(f"| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |\n")
        f.write(f"| Win Rate | {metrics.get('win_rate', 0):.2f}% |\n")
        f.write(f"| Profit Factor | {metrics.get('profit_factor', 0):.2f} |\n")
        f.write(f"| Total Trades | {metrics.get('num_trades', 0)} |\n")
        
        # Add more detailed metrics as needed
        
    logger.info(f"Generated backtest report in {output_dir}")