import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .backtest_engine import BacktestEngine
from .strategies import get_strategy
from .data_provider import DataProvider
from .risk_manager import RiskManager
from .logger import logger

class BacktestOrchestrator:
    """
    Orchestrates the backtesting process, including strategy execution,
    risk management, and performance analysis.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_provider: DataProvider,
        output_dir: str = "backtest_results"
    ):
        """
        Initialize the backtest orchestrator.
        
        Args:
            config: Configuration dictionary
            data_provider: Data provider instance
            output_dir: Directory to save backtest results
        """
        self.config = config
        self.data_provider = data_provider
        self.output_dir = Path(output_dir)
        self.results = {}
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def run(self) -> Dict[str, Any]:
        """
        Run the backtest with the configured strategy and parameters.
        
        Returns:
            Dictionary containing backtest results
        """
        logger.info("Starting backtest...")
        start_time = datetime.now()
        
        try:
            # Initialize components
            strategy = self._initialize_strategy()
            risk_manager = RiskManager(self.config.get('risk', {}))
            
            # Initialize backtest engine
            engine = BacktestEngine(
                strategy=strategy,
                data_provider=self.data_provider,
                risk_manager=risk_manager,
                config=self.config
            )
            
            # Run backtest
            results = await engine.run()
            
            # Save results
            self._save_results(results)
            
            # Generate reports
            self._generate_reports(results)
            
            # Calculate and log performance metrics
            self._log_performance_metrics(results)
            
            logger.info(f"Backtest completed in {datetime.now() - start_time}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            raise
            
    def _initialize_strategy(self):
        """Initialize the trading strategy."""
        strategy_config = self.config.get('strategy', {})
        strategy_name = strategy_config.get('name')
        params = strategy_config.get('params', {})
        
        if not strategy_name:
            raise ValueError("No strategy specified in config")
            
        return get_strategy(strategy_name, **params)
        
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save backtest results to disk."""
        try:
            # Save raw results
            results_file = self.output_dir / 'backtest_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            # Save trades to CSV if available
            if 'trades' in results and results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_file = self.output_dir / 'trades.csv'
                trades_df.to_csv(trades_file, index=False)
                
            logger.info(f"Backtest results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
            
    def _generate_reports(self, results: Dict[str, Any]) -> None:
        """Generate performance reports and visualizations."""
        try:
            # Generate equity curve plot
            self._plot_equity_curve(results)
            
            # Generate performance metrics report
            self._generate_metrics_report(results)
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            
    def _plot_equity_curve(self, results: Dict[str, Any]) -> None:
        """Plot the equity curve."""
        try:
            import matplotlib.pyplot as plt
            
            equity = results.get('equity_curve', [])
            if not equity:
                return
                
            plt.figure(figsize=(12, 6))
            plt.plot([e['timestamp'] for e in equity], [e['equity'] for e in equity])
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            
            # Save the plot
            plot_file = self.output_dir / 'equity_curve.png'
            plt.savefig(plot_file)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping equity curve plot")
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            
    def _generate_metrics_report(self, results: Dict[str, Any]) -> None:
        """Generate a text report of performance metrics."""
        try:
            metrics = results.get('metrics', {})
            if not metrics:
                return
                
            report = "# Backtest Performance Report\n\n"
            report += f"## Strategy: {self.config.get('strategy', {}).get('name', 'Unknown')}\n"
            report += f"## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add key metrics
            report += "## Key Metrics\n"
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    report += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
                else:
                    report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                    
            # Save report
            report_file = self.output_dir / 'performance_report.md'
            with open(report_file, 'w') as f:
                f.write(report)
                
        except Exception as e:
            logger.error(f"Error generating metrics report: {e}")
            
    def _log_performance_metrics(self, results: Dict[str, Any]) -> None:
        """Log key performance metrics."""
        metrics = results.get('metrics', {})
        if not metrics:
            return
            
        logger.info("\n" + "="*50)
        logger.info("BACKTEST PERFORMANCE METRICS")
        logger.info("="*50)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key.replace('_', ' ').title():<30}: {value:>10.4f}")
            else:
                logger.info(f"{key.replace('_', ' ').title():<30}: {value}")
                
        logger.info("="*50)