"""
Advanced backtesting module with enhanced metrics and walk-forward analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
from backtest_engine import BacktestEngine, BacktestResult
from advanced_metrics import AdvancedMetrics, create_trade_analysis_report

class EnhancedBacktest(BacktestEngine):
    """Enhanced backtesting engine with advanced metrics and walk-forward analysis."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.advanced_metrics = AdvancedMetrics()
    
    async def _calculate_metrics(self, trades: List[Dict[str, Any]], 
                              initial_balance: float = 10000.0) -> Dict[str, Any]:
        """
        Calculate advanced metrics for the backtest results.
        
        Args:
            trades: List of trade dictionaries
            initial_balance: Starting balance for the backtest
            
        Returns:
            Dictionary containing all calculated metrics
        """
        # Convert trades to the format expected by AdvancedMetrics
        formatted_trades = []
        for trade in trades:
            formatted_trades.append({
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'size': trade['size'],
                'side': trade['side'].lower(),
                'pnl': trade['pnl'],
                'pnl_pct': trade['pnl_pct']
            })
        
        # Calculate metrics
        metrics = self.advanced_metrics.calculate_all_metrics(formatted_trades, initial_balance)
        
        # Generate comprehensive report
        report = create_trade_analysis_report(formatted_trades, initial_balance)
        
        return {
            **metrics,
            'report': report
        }
    
    async def run_walk_forward_analysis(
        self,
        initial_train_period: int = 180,  # days
        test_period: int = 30,  # days
        step: int = 30,  # days
        initial_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis on the strategy.
        
        Args:
            initial_train_period: Initial training period in days
            test_period: Test period in days
            step: Step size for moving the window forward in days
            initial_balance: Starting balance for the backtest
            
        Returns:
            Dictionary containing walk-forward analysis results
        """
        # Get all available data
        data = await self.data_provider.get_historical_data(
            self.symbol,
            self.timeframe,
            self.start_date,
            self.end_date
        )
        
        if data.empty:
            raise ValueError("No data available for the specified period")
        
        # Convert dates to timestamps for easier manipulation
        start_ts = pd.Timestamp(self.start_date)
        end_ts = pd.Timestamp(self.end_date)
        
        walk_forward_results = []
        
        current_train_start = start_ts
        
        while current_train_start + pd.Timedelta(days=initial_train_period + test_period) <= end_ts:
            train_end = current_train_start + pd.Timedelta(days=initial_train_period)
            test_end = train_end + pd.Timedelta(days=test_period)
            
            # Run backtest on training period
            self.start_date = current_train_start.to_pydatetime()
            self.end_date = train_end.to_pydatetime()
            
            train_result = await self.run()
            
            # Run backtest on test period with the trained model
            self.start_date = train_end.to_pydatetime()
            self.end_date = test_end.to_pydatetime()
            
            test_result = await self.run()
            
            walk_forward_results.append({
                'train_period': (current_train_start, train_end),
                'test_period': (train_end, test_end),
                'train_metrics': train_result.metrics,
                'test_metrics': test_result.metrics,
                'train_result': train_result,
                'test_result': test_result
            })
            
            # Move the window forward
            current_train_start += pd.Timedelta(days=step)
        
        # Calculate walk-forward metrics
        test_returns = [r['test_metrics']['total_return'] for r in walk_forward_results]
        train_returns = [r['train_metrics']['total_return'] for r in walk_forward_results]
        
        walk_forward_metrics = {
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'avg_train_return': np.mean(train_returns),
            'std_train_return': np.std(train_returns),
            'consistency': np.corrcoef(train_returns, test_returns)[0, 1],
            'num_periods': len(walk_forward_results)
        }
        
        return {
            'walk_forward_metrics': walk_forward_metrics,
            'period_results': walk_forward_results,
            'train_returns': train_returns,
            'test_returns': test_returns
        }
    
    def generate_report(
        self, 
        result: Dict[str, Any], 
        output_dir: str = 'reports'
    ) -> str:
        """
        Generate a comprehensive HTML report of the backtest results.
        
        Args:
            result: Backtest result dictionary
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / f'backtest_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        # Create the HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ width: 90%; margin: 0 auto; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ background: #f4f4f4; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 1.5em; font-weight: bold; }}
                .chart {{ margin: 30px 0; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Backtest Report</h1>
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-label">Total Return</div>
                            <div class="metric-value">{total_return:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Sharpe Ratio</div>
                            <div class="metric-value">{sharpe_ratio:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Max Drawdown</div>
                            <div class="metric-value">{max_drawdown:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Win Rate</div>
                            <div class="metric-value">{win_rate:.2f}%</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Equity Curve</h2>
                    <div class="chart" id="equity-chart">[Equity Curve Chart]</div>
                </div>
                
                <div class="section">
                    <h2>Trade Analysis</h2>
                    <pre>{trade_analysis}</pre>
                </div>
            </div>
        </body>
        </html>
        """.format(
            total_return=result['metrics'].get('total_return', 0) * 100,
            sharpe_ratio=result['metrics'].get('sharpe_ratio', 0),
            max_drawdown=abs(result['metrics'].get('max_drawdown', 0)) * 100,
            win_rate=result['metrics'].get('win_rate', 0) * 100,
            trade_analysis=result.get('report', 'No trade analysis available')
        )
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        return str(report_path.absolute())
