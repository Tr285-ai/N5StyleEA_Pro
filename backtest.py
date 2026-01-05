import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import warnings
from scipy import stats
from joblib import Parallel, delayed
import random
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('backtest')

class BacktestEngine:
    """
    Advanced backtesting engine with walk-forward analysis and Monte Carlo simulation.
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize the backtesting engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.strategy = self._load_strategy()
        self.results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {},
            'walk_forward': {},
            'monte_carlo': {}
        }
        self.initial_balance = 0
        self.commission = 0
        self.slippage = 0
        self.risk_free_rate = 0.01  # 1% annual risk-free rate
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    def _load_strategy(self):
        """Load trading strategy based on config."""
        try:
            strategy_name = self.config.get('strategy', {}).get('name', 'MovingAverageCrossover')
            strategy_params = self.config.get('strategy', {}).get('params', {})
            
            # Dynamically import the strategy class
            module = __import__(f'strategies.{strategy_name.lower()}', fromlist=[strategy_name])
            strategy_class = getattr(module, strategy_name)
            
            return strategy_class(strategy_params)
            
        except Exception as e:
            logger.error(f"Error loading strategy: {str(e)}")
            raise
    
    def _calculate_drawdowns(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics from equity curve."""
        # Convert to pandas Series if not already
        if not isinstance(equity_curve, pd.Series):
            equity_curve = pd.Series(equity_curve)
            
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Calculate max drawdown
        max_drawdown = drawdown.min() * 100  # as percentage
        
        # Calculate drawdown duration
        drawdown_duration = (drawdown != 0).astype(int)
        drawdown_duration = drawdown_duration.groupby(
            (drawdown_duration != drawdown_duration.shift()).cumsum()
        ).cumsum()
        
        max_drawdown_duration = drawdown_duration.max()
        
        return {
            'max_drawdown_pct': abs(max_drawdown),
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def _calculate_metrics(self, trades: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if trades.empty or len(equity_curve) < 2:
            return {}
            
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annualized_return = (1 + total_return/100) ** (252/len(equity_curve)) - 1
        annualized_vol = equity_curve.pct_change().std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / (annualized_vol + 1e-9)
        
        # Trade metrics
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = winning_trades['pnl_pct'].mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades['pnl_pct'].mean()) if not losing_trades.empty else 0
        profit_factor = winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if not losing_trades.empty else float('inf')
        
        # Risk metrics
        drawdowns = self._calculate_drawdowns(equity_curve)
        max_drawdown = drawdowns['max_drawdown_pct']
        max_drawdown_duration = drawdowns['max_drawdown_duration']
        
        # Calculate Calmar ratio
        calmar_ratio = (annualized_return * 100) / (max_drawdown + 1e-9)
        
        # Calculate Sortino ratio
        downside_returns = equity_curve.pct_change()[equity_curve.pct_change() < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.risk_free_rate) / (downside_volatility + 1e-9)
        
        # Calculate recovery factor
        recovery_factor = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / abs(equity_curve.min() - equity_curve.iloc[0]) if (equity_curve.min() < equity_curve.iloc[0]) else float('inf')
        
        # Calculate expectancy
        avg_win_trade = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss_trade = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        win_loss_ratio = abs(avg_win_trade / (avg_loss_trade + 1e-9))
        expectancy = (win_rate/100 * avg_win_trade) - ((100 - win_rate)/100 * abs(avg_loss_trade))
        
        return {
            # Returns
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return * 100,
            'monthly_return_pct': ((1 + total_return/100) ** (1/(len(equity_curve)/21)) - 1) * 100,
            
            # Risk
            'annualized_volatility_pct': annualized_vol * 100,
            'max_drawdown_pct': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            
            # Ratios
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,
            
            # Trade statistics
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy,
            
            # Risk-adjusted returns
            'risk_adjusted_return': total_return / (max_drawdown + 1e-9),
            'k_ratio': self._calculate_k_ratio(equity_curve),
            'tail_ratio': self._calculate_tail_ratio(equity_curve),
            
            # Other metrics
            'profit_to_max_drawdown': total_return / (max_drawdown + 1e-9),
            'avg_trade_duration': (trades['exit_time'] - trades['entry_time']).mean().total_seconds() / 3600 if 'exit_time' in trades.columns and 'entry_time' in trades.columns else 0
        }
    
    def _calculate_k_ratio(self, equity_curve: pd.Series, lookback: int = 20) -> float:
        """Calculate K-Ratio (performance consistency)."""
        if len(equity_curve) < lookback + 1:
            return 0.0
            
        returns = equity_curve.pct_change().dropna()
        if len(returns) < lookback + 1:
            return 0.0
            
        # Calculate rolling returns
        rolling_returns = returns.rolling(window=lookback).apply(lambda x: (1 + x).prod() - 1)
        
        # Calculate K-Ratio
        x = np.arange(len(rolling_returns))
        y = rolling_returns.values
        slope, _, _, _, _ = stats.linregress(x[~np.isnan(y)], y[~np.isnan(y)])
        
        return slope * 100  # as percentage
    
    def _calculate_tail_ratio(self, equity_curve: pd.Series, percentile: float = 5) -> float:
        """Calculate Tail Ratio (good vs bad volatility)."""
        returns = equity_curve.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
            
        # Calculate upside and downside returns
        upside = returns[returns >= 0]
        downside = returns[returns < 0]
        
        if len(upside) == 0 or len(downside) == 0:
            return 0.0
            
        # Calculate average of top and bottom percentiles
        avg_upside = upside.mean()
        avg_downside = abs(downside.mean())
        
        return avg_upside / (avg_downside + 1e-9)
    
    def _generate_report(self, results: Dict, output_dir: str = 'reports') -> None:
        """Generate comprehensive backtest report."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save trades to CSV
            trades_df = pd.DataFrame(results['trades'])
            trades_csv = os.path.join(output_dir, f'trades_{timestamp}.csv')
            trades_df.to_csv(trades_csv, index=False)
            
            # Save equity curve to CSV
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_csv = os.path.join(output_dir, f'equity_curve_{timestamp}.csv')
            equity_df.to_csv(equity_csv, index=False)
            
            # Generate metrics report
            metrics = results['metrics']
            report = f"""
            ===== BACKTEST REPORT =====
            Date: {datetime.now()}
            Strategy: {self.strategy.__class__.__name__}
            Symbol: {self.symbol}
            Period: {self.start_date} to {self.end_date}
            Initial Balance: ${self.initial_balance:,.2f}
            Final Balance: ${equity_df['equity'].iloc[-1]:,.2f}
            {'='*80}
            
            [PERFORMANCE METRICS]
            Total Return: {metrics['total_return_pct']:.2f}%
            Annualized Return: {metrics['annualized_return_pct']:.2f}%
            Monthly Return: {metrics['monthly_return_pct']:.2f}%
            
            [RISK METRICS]
            Max Drawdown: {metrics['max_drawdown_pct']:.2f}%
            Max Drawdown Duration: {metrics['max_drawdown_duration']} periods
            Annualized Volatility: {metrics['annualized_volatility_pct']:.2f}%
            
            [RISK-ADJUSTED RETURNS]
            Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
            Sortino Ratio: {metrics['sortino_ratio']:.2f}
            Calmar Ratio: {metrics['calmar_ratio']:.2f}
            K-Ratio: {metrics['k_ratio']:.4f}
            Tail Ratio: {metrics['tail_ratio']:.2f}
            
            [TRADE STATISTICS]
            Total Trades: {metrics['total_trades']}
            Win Rate: {metrics['win_rate_pct']:.2f}%
            Avg Win: {metrics['avg_win_pct']:.2f}%
            Avg Loss: {metrics['avg_loss_pct']:.2f}%
            Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}
            Profit Factor: {metrics['profit_factor']:.2f}
            Expectancy: ${metrics['expectancy']:.2f} per trade
            
            [FILES]
            Trades CSV: {os.path.abspath(trades_csv)}
            Equity Curve CSV: {os.path.abspath(equity_csv)}
            """
            
            # Save report to file
            report_file = os.path.join(output_dir, f'report_{timestamp}.txt')
            with open(report_file, 'w') as f:
                f.write(report)
                
            # Print report to console
            print(report)
            
            # Generate plots
            self._generate_plots(results, output_dir, timestamp)
            
            logger.info(f"Backtest report saved to {os.path.abspath(report_file)}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_plots(self, results: Dict, output_dir: str, timestamp: str) -> None:
        """Generate performance plots."""
        try:
            # Set style
            plt.style.use('seaborn')
            sns.set_palette("husl")
            
            # Plot equity curve
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Equity curve
            equity = pd.DataFrame(results['equity_curve'])
            equity['timestamp'] = pd.to_datetime(equity['timestamp'])
            equity.set_index('timestamp', inplace=True)
            
            # Plot equity
            ax1.plot(equity.index, equity['equity'], label='Equity', linewidth=2)
            ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Equity ($)', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # Plot drawdown
            rolling_max = equity['equity'].cummax()
            drawdown = (equity['equity'] - rolling_max) / rolling_max * 100
            ax2.fill_between(equity.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            
            # Format x-axis
            date_format = DateFormatter("%Y-%m-%d")
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(date_format)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save figure
            plot_file = os.path.join(output_dir, f'equity_curve_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot monthly returns heatmap
            self._plot_monthly_returns(equity, output_dir, timestamp)
            
            # Plot trade distribution
            self._plot_trade_distribution(pd.DataFrame(results['trades']), output_dir, timestamp)
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise
    
    def _plot_monthly_returns(self, equity: pd.DataFrame, output_dir: str, timestamp: str) -> None:
        """Generate monthly returns heatmap."""
        try:
            # Calculate monthly returns
            monthly_returns = equity['equity'].resample('M').last().pct_change().dropna()
            monthly_returns = monthly_returns * 100  # Convert to percentage
            
            # Create pivot table for heatmap
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_returns_df = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month_name().str[:3],
                'return': monthly_returns.values
            })
            
            # Pivot for heatmap
            returns_pivot = monthly_returns_df.pivot(
                index='year', 
                columns='month', 
                values='return'
            )
            
            # Reorder columns to be Jan-Dec
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            returns_pivot = returns_pivot[month_order]
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                returns_pivot, 
                annot=True, 
                fmt=".2f", 
                cmap='RdYlGn', 
                center=0,
                linewidths=0.5,
                annot_kws={"size": 8},
                cbar_kws={'label': 'Monthly Return (%)'}
            )
            plt.title('Monthly Returns (%)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            plot_file = os.path.join(output_dir, f'monthly_returns_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate monthly returns heatmap: {str(e)}")
    
    def _plot_trade_distribution(self, trades: pd.DataFrame, output_dir: str, timestamp: str) -> None:
        """Generate trade distribution plots."""
        try:
            if trades.empty:
                return
                
            # Convert to DataFrame if not already
            if not isinstance(trades, pd.DataFrame):
                trades = pd.DataFrame(trades)
                
            # Ensure required columns exist
            if 'pnl' not in trades.columns:
                return
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
            
            # P&L Distribution
            sns.histplot(trades['pnl'], kde=True, ax=axes[0, 0])
            axes[0, 0].axvline(trades['pnl'].mean(), color='r', linestyle='--', label=f'Mean: ${trades["pnl"].mean():.2f}')
            axes[0, 0].set_title('P&L Distribution', fontsize=12)
            axes[0, 0].set_xlabel('P&L ($)')
            axes[0, 0].legend()
            
            # Win/Loss Pie Chart
            win_loss = trades['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss').value_counts()
            axes[0, 1].pie(
                win_loss, 
                labels=win_loss.index, 
                autopct='%1.1f%%',
                startangle=90,
                colors=['#4CAF50', '#F44336']
            )
            axes[0, 1].set_title('Win/Loss Ratio', fontsize=12)
            
            # P&L Over Time
            if 'exit_time' in trades.columns:
                trades['exit_time'] = pd.to_datetime(trades['exit_time'])
                trades = trades.sort_values('exit_time')
                trades['cum_pnl'] = trades['pnl'].cumsum()
                axes[1, 0].plot(trades['exit_time'], trades['cum_pnl'], marker='o', markersize=4)
                axes[1, 0].set_title('Cumulative P&L Over Time', fontsize=12)
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Cumulative P&L ($)')
                axes[1, 0].grid(True, linestyle='--', alpha=0.7)
                
                # Format x-axis
                date_format = DateFormatter("%Y-%m-%d")
                axes[1, 0].xaxis.set_major_formatter(date_format)
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # Trade Duration Histogram
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                trades['duration'] = (pd.to_datetime(trades['exit_time']) - 
                                     pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 3600
                sns.histplot(trades['duration'], bins=20, kde=True, ax=axes[1, 1])
                axes[1, 1].set_title('Trade Duration (Hours)', fontsize=12)
                axes[1, 1].set_xlabel('Duration (hours)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure
            plot_file = os.path.join(output_dir, f'trade_analysis_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate trade distribution plots: {str(e)}")
    
    def _run_walk_forward_analysis(self, data: pd.DataFrame, 
                                 initial_balance: float, 
                                 commission: float) -> Dict:
        """
        Perform walk-forward analysis on the backtest.
        
        Args:
            data: Historical price data
            initial_balance: Starting account balance
            commission: Commission per trade
            
        Returns:
            Dict: Walk-forward analysis results
        """
        try:
            # Define walk-forward parameters
            total_bars = len(data)
            in_sample_ratio = 0.7  # 70% in-sample, 30% out-of-sample
            in_sample_bars = int(total_bars * in_sample_ratio)
            
            # Split data into in-sample and out-of-sample
            in_sample_data = data.iloc[:in_sample_bars]
            out_sample_data = data.iloc[in_sample_bars:]
            
            # Run in-sample backtest
            logger.info("Running in-sample backtest...")
            in_sample_results = self._run_backtest(
                in_sample_data, initial_balance, commission, is_walk_forward=True
            )
            
            # Run out-of-sample backtest
            logger.info("Running out-of-sample backtest...")
            out_sample_results = self._run_backtest(
                out_sample_data, initial_balance, commission, is_walk_forward=True
            )
            
            # Calculate walk-forward efficiency
            is_return = in_sample_results['metrics']['total_return_pct']
            oos_return = out_sample_results['metrics']['total_return_pct']
            wfe = (oos_return / (abs(is_return) + 1e-9)) * 100 if is_return != 0 else 0
            
            return {
                'in_sample': in_sample_results,
                'out_of_sample': out_sample_results,
                'walk_forward_efficiency': wfe,
                'is_oos_correlation': np.corrcoef(
                    [t['pnl_pct'] for t in in_sample_results['trades']],
                    [t['pnl_pct'] for t in out_sample_results['trades']]
                )[0, 1] if in_sample_results['trades'] and out_sample_results['trades'] else 0
            }
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {str(e)}")
            return {}
    
    def _run_monte_carlo_simulation(self, trades: List[Dict], 
                                  initial_balance: float,
                                  num_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation on trade sequence.
        
        Args:
            trades: List of trade results
            initial_balance: Starting account balance
            num_simulations: Number of simulations to run
            
        Returns:
            Dict: Monte Carlo simulation results
        """
        try:
            if not trades:
                return {}
                
            # Extract P&L percentages
            pnl_pct = [t['pnl_pct'] for t in trades if 'pnl_pct' in t]
            if not pnl_pct:
                return {}
                
            # Run simulations
            simulation_results = []
            for _ in range(num_simulations):
                # Shuffle trades
                random.shuffle(pnl_pct)
                
                # Calculate equity curve
                equity = [initial_balance]
                for pct in pnl_pct:
                    equity.append(equity[-1] * (1 + pct/100))
                
                # Calculate metrics
                equity_series = pd.Series(equity)
                returns = equity_series.pct_change().dropna()
                total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
                max_drawdown = self._calculate_drawdowns(equity_series)['max_drawdown_pct']
                
                simulation_results.append({
                    'equity': equity_series,
                    'total_return_pct': total_return,
                    'max_drawdown_pct': max_drawdown,
                    'sharpe_ratio': (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
                })
            
            # Calculate statistics
            total_returns = [s['total_return_pct'] for s in simulation_results]
            max_drawdowns = [s['max_drawdown_pct'] for s in simulation_results]
            sharpe_ratios = [s['sharpe_ratio'] for s in simulation_results]
            
            # Calculate confidence intervals
            def confidence_interval(data, confidence=0.95):
                a = 1.0 * np.array(data)
                n = len(a)
                m, se = np.mean(a), stats.sem(a)
                h = se * stats.t.ppf((1 + confidence) / 2., n-1)
                return m - h, m + h
                
            ci_low, ci_high = confidence_interval(total_returns)
            
            return {
                'num_simulations': num_simulations,
                'mean_return_pct': np.mean(total_returns),
                'median_return_pct': np.median(total_returns),
                'std_return_pct': np.std(total_returns),
                'min_return_pct': np.min(total_returns),
                'max_return_pct': np.max(total_returns),
                'confidence_interval_95_pct': (ci_low, ci_high),
                'prob_profit': len([r for r in total_returns if r > 0]) / num_simulations * 100,
                'avg_max_drawdown_pct': np.mean(max_drawdowns),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'simulation_results': simulation_results
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {}
    
    def _optimize_parameters(self, data: pd.DataFrame, 
                           param_grid: Dict[str, List[Any]],
                           initial_balance: float,
                           commission: float) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: Historical price data
            param_grid: Dictionary of parameter names and lists of values to try
            initial_balance: Starting account balance
            commission: Commission per trade
            
        Returns:
            Dict: Optimization results with best parameters and metrics
        """
        try:
            from sklearn.model_selection import ParameterGrid
            
            best_params = None
            best_metric = -float('inf')
            results = []
            
            # Generate all parameter combinations
            param_combinations = list(ParameterGrid(param_grid))
            total_combinations = len(param_combinations)
            
            logger.info(f"Testing {total_combinations} parameter combinations...")
            
            for i, params in enumerate(param_combinations, 1):
                try:
                    # Update strategy with current parameters
                    self.strategy = self.strategy.__class__(params)
                    
                    # Run backtest
                    result = self._run_backtest(data, initial_balance, commission, is_optimization=True)
                    
                    # Calculate optimization metric (e.g., Sharpe ratio)
                    metric = result['metrics'].get('sharpe_ratio', 0)
                    
                    # Store results
                    result_entry = {
                        'params': params,
                        'metric': metric,
                        'total_return_pct': result['metrics'].get('total_return_pct', 0),
                        'max_drawdown_pct': result['metrics'].get('max_drawdown_pct', 0),
                        'win_rate_pct': result['metrics'].get('win_rate_pct', 0)
                    }
                    results.append(result_entry)
                    
                    # Update best parameters
                    if metric > best_metric:
                        best_metric = metric
                        best_params = params
                    
                    logger.info(f"Tested {i}/{total_combinations} - Metric: {metric:.4f} - Params: {params}")
                    
                except Exception as e:
                    logger.error(f"Error testing parameters {params}: {str(e)}")
                    continue
            
            # Sort results by metric
            results.sort(key=lambda x: x['metric'], reverse=True)
            
            return {
                'best_params': best_params,
                'best_metric': best_metric,
                'all_results': results,
                'top_10_results': results[:10]
            }
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            return {}
    
    def _run_backtest(self, data: pd.DataFrame, 
                     initial_balance: float,
                     commission: float,
                     is_walk_forward: bool = False,
                     is_optimization: bool = False) -> Dict:
        """
        Core backtest logic.
        
        Args:
            data: Historical price data
            initial_balance: Starting account balance
            commission: Commission per trade
            is_walk_forward: Whether this is part of walk-forward analysis
            is_optimization: Whether this is part of parameter optimization
            
        Returns:
            Dict: Backtest results
        """
        try:
            # Initialize variables
            balance = initial_balance
            position = 0
            entry_price = 0
            entry_time = None
            trade_count = 0
            wins = 0
            losses = 0
            max_drawdown = 0
            peak_balance = initial_balance
            equity_curve = []
            trades = []
            
            # Convert DataFrame to list of dictionaries for iteration
            data_dict = data.to_dict('records')
            
            # Main backtest loop
            for i in range(1, len(data_dict)):
                current = data_dict[i]
                previous = data_dict[i-1]
                
                # Prepare data for strategy
                historical_data = {
                    'open': [x['open'] for x in data_dict[:i+1]],
                    'high': [x['high'] for x in data_dict[:i+1]],
                    'low': [x['low'] for x in data_dict[:i+1]],
                    'close': [x['close'] for x in data_dict[:i+1]],
                    'volume': [x.get('volume', 0) for x in data_dict[:i+1]],
                    'timestamp': [x.get('timestamp') for x in data_dict[:i+1]]
                }
                
                # Get signal from strategy
                signal = self.strategy.update(
                    self.symbol, 
                    historical_data,
                    account_balance=balance
                )
                
                # Process signal if we have one
                if signal:
                    direction = signal.get('direction')
                    price = signal.get('price', current['close'])
                    position_size = signal.get('position_size')
                    
                    # Close existing position if any
                    if position != 0:
                        # Calculate P&L
                        if position > 0:  # Long position
                            pnl = (price - entry_price) * position
                        else:  # Short position
                            pnl = (entry_price - price) * abs(position)
                            
                        # Apply commission
                        pnl -= abs(position * price * commission)
                        
                        # Track win/loss
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                            
                        # Record trade
                        trade = {
                            'entry_time': entry_time,
                            'exit_time': current.get('timestamp', i),
                            'direction': 'LONG' if position > 0 else 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': price,
                            'position_size': abs(position),
                            'pnl': pnl,
                            'pnl_pct': (pnl / (abs(position) * entry_price)) * 100 if entry_price > 0 else 0,
                            'balance': balance + pnl,
                            'duration': (current.get('timestamp', i) - entry_time).total_seconds() / 3600 
                                      if entry_time and 'timestamp' in current else 0
                        }
                        trades.append(trade)
                        
                        # Update balance
                        balance += pnl
                            
                        # Reset position
                        position = 0
                        entry_price = 0
                        entry_time = None
                    
                    # Open new position if we have a direction and position size
                    if direction and position_size and position_size > 0:
                        if direction.upper() == 'BUY':
                            position = position_size
                            entry_price = price
                            entry_time = current.get('timestamp', i)
                        elif direction.upper() == 'SELL':
                            position = -position_size
                            entry_price = price
                            entry_time = current.get('timestamp', i)
                
                # Update equity curve
                if position != 0:
                    current_value = position * current['close']
                    equity = balance + current_value
                else:
                    equity = balance
                    
                equity_curve.append({
                    'timestamp': current.get('timestamp', i),
                    'equity': equity,
                    'price': current['close'],
                    'position': position
                })
                
                # Update max drawdown
                peak_balance = max(peak_balance, equity)
                drawdown = (peak_balance - equity) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)
                
            # Close any open position at the end
            if position != 0:
                if position > 0:  # Long position
                    pnl = (current['close'] - entry_price) * position
                else:  # Short position
                    pnl = (entry_price - current['close']) * abs(position)
                    
                pnl -= abs(position * current['close'] * commission)
                balance += pnl
                
                # Track win/loss
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                    
                # Record final trade
                trade = {
                    'entry_time': entry_time,
                    'exit_time': current.get('timestamp', len(data_dict) - 1),
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': current['close'],
                    'position_size': abs(position),
                    'pnl': pnl,
                    'pnl_pct': (pnl / (abs(position) * entry_price)) * 100 if entry_price > 0 else 0,
                    'balance': balance,
                    'duration': (current.get('timestamp', len(data_dict) - 1) - entry_time).total_seconds() / 3600 
                              if entry_time else 0
                }
                trades.append(trade)
                
                # Update equity curve
                equity = balance
                equity_curve.append({
                    'timestamp': current.get('timestamp', len(data_dict) - 1),
                    'equity': equity,
                    'price': current['close'],
                    'position': 0
                })
                
            # Calculate performance metrics
            equity_series = pd.Series([x['equity'] for x in equity_curve])
            metrics = self._calculate_metrics(pd.DataFrame(trades), equity_series)
            
            # For optimization, return minimal results
            if is_optimization:
                return {
                    'metrics': metrics,
                    'trades': trades[:10],  # Return only first 10 trades to save memory
                    'equity_curve': equity_curve[::max(1, len(equity_curve)//1000)]  # Sample equity curve
                }
                
            # For walk-forward, return full results but without additional analysis
            if is_walk_forward:
                return {
                    'trades': trades,
                    'equity_curve': equity_curve,
                    'metrics': metrics
                }
                
            # For standard backtest, return full results with additional analysis
            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'metrics': metrics,
                'walk_forward': self._run_walk_forward_analysis(data, initial_balance, commission) if not is_walk_forward else None,
                'monte_carlo': self._run_monte_carlo_simulation(trades, initial_balance) if trades else None
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise
    
    def run(self, symbol: str, start_date: str, end_date: str, 
            initial_balance: float = 10000.0, 
            commission: float = 0.001,
            slippage: float = 0.0001,
            run_walk_forward: bool = True,
            run_monte_carlo: bool = True,
            monte_carlo_simulations: int = 1000,
            optimize_params: bool = False,
            param_grid: Optional[Dict] = None) -> Dict:
        """
        Run backtest with the given parameters.
        
        Args:
            symbol: Trading symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            initial_balance: Starting account balance
            commission: Commission per trade (percentage)
            slippage: Slippage per trade (percentage)
            run_walk_forward: Whether to run walk-forward analysis
            run_monte_carlo: Whether to run Monte Carlo simulation
            monte_carlo_simulations: Number of Monte Carlo simulations
            optimize_params: Whether to optimize parameters
            param_grid: Parameter grid for optimization
            
        Returns:
            Dict: Backtest results
        """
        try:
            self.symbol = symbol
            self.start_date = start_date
            self.end_date = end_date
            self.initial_balance = initial_balance
            self.commission = commission
            self.slippage = slippage
            
            logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
            
            # Load historical data
            data = self.load_data(symbol, start_date, end_date)
            
            # Run parameter optimization if requested
            if optimize_params and param_grid:
                logger.info("Starting parameter optimization...")
                optimization_results = self._optimize_parameters(
                    data, param_grid, initial_balance, commission
                )
                
                if optimization_results and 'best_params' in optimization_results:
                    logger.info(f"Best parameters: {optimization_results['best_params']}")
                    # Update strategy with best parameters
                    self.strategy = self.strategy.__class__(optimization_results['best_params'])
                else:
                    logger.warning("Parameter optimization did not return valid results")
            
            # Run main backtest
            results = self._run_backtest(
                data, 
                initial_balance, 
                commission,
                is_walk_forward=False
            )
            
            # Add additional analysis if not in optimization mode
            if not optimize_params:
                # Run walk-forward analysis if requested
                if run_walk_forward and len(data) > 100:  # Need enough data for walk-forward
                    logger.info("Running walk-forward analysis...")
                    results['walk_forward'] = self._run_walk_forward_analysis(
                        data, initial_balance, commission
                    )
                else:
                    logger.info("Skipping walk-forward analysis (not enough data or disabled)")
                
                # Run Monte Carlo simulation if requested and we have trades
                if run_monte_carlo and results.get('trades'):
                    logger.info("Running Monte Carlo simulation...")
                    results['monte_carlo'] = self._run_monte_carlo_simulation(
                        results['trades'], 
                        initial_balance,
                        num_simulations=monte_carlo_simulations
                    )
                else:
                    logger.info("Skipping Monte Carlo simulation (no trades or disabled)")
            
            # Generate report
            self._generate_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            raise
    
    def load_data(self, symbol: str, start_date: str, end_date: str, 
                 timeframe: str = '1h') -> pd.DataFrame:
        """
        Load historical market data.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # In a real implementation, this would load from your data source
            # For now, we'll create sample data
            date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
            n = len(date_range)
            
            # Generate random walk for prices
            np.random.seed(42)
            returns = np.random.normal(0.0001, 0.01, n)
            prices = 100.0 * (1 + np.cumsum(returns))
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': date_range,
                'open': prices,
                'high': prices + np.random.uniform(0, 0.5, n),
                'low': prices - np.random.uniform(0, 0.5, n),
                'close': prices + np.random.uniform(-0.25, 0.25, n),
                'volume': np.random.randint(100, 1000, n)
            })
            
            # Ensure high >= close >= low
            df['high'] = df[['high', 'close']].max(axis=1)
            df['low'] = df[['low', 'close']].min(axis=1)
            df['high'] = df[['high', 'open']].max(axis=1)
            df['low'] = df[['low', 'open']].min(axis=1)
            
            # Resample to desired timeframe
            if timeframe != '1h':
                df = df.resample(timeframe, on='timestamp').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    backtester = BacktestEngine('config.json')
    
    # Define parameter grid for optimization (optional)
    param_grid = {
        'fast_ma': [5, 10, 15],
        'slow_ma': [20, 30, 50],
        'rsi_period': [7, 14, 21],
        'atr_multiplier': [1.5, 2.0, 2.5]
    }
    
    # Run backtest with optimization
    results = backtester.run(
        symbol='EURUSD',
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_balance=10000.0,
        commission=0.001,
        slippage=0.0001,
        run_walk_forward=True,
        run_monte_carlo=True,
        monte_carlo_simulations=1000,
        optimize_params=True,  # Set to True to enable parameter optimization
        param_grid=param_grid  # Pass parameter grid for optimization
    )