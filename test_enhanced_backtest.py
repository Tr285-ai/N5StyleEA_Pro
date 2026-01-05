import asyncio
import pandas as pd
from datetime import datetime, timedelta
from advanced_backtest import EnhancedBacktest
from strategies.moving_average_crossover import MovingAverageCrossover
from data_provider import DataProvider

async def main():
    # Initialize data provider (replace with your actual data provider)
    data_provider = DataProvider()
    
    # Initialize strategy
    strategy = MovingAverageCrossover(
        short_window=10,
        long_window=30
    )
    
    # Initialize enhanced backtest
    backtest = EnhancedBacktest(
        strategy=strategy,
        data_provider=data_provider,
        symbol='BTC/USDT',
        timeframe='1d',
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 1, 1),
        initial_balance=10000.0
    )
    
    # Run standard backtest
    print("Running standard backtest...")
    result = await backtest.run()
    print(f"Standard backtest completed. Total return: {result.metrics['total_return']*100:.2f}%")
    
    # Generate report
    report_path = backtest.generate_report(result.to_dict())
    print(f"Report generated at: {report_path}")
    
    # Run walk-forward analysis
    print("\nRunning walk-forward analysis...")
    wf_results = await backtest.run_walk_forward_analysis(
        initial_train_period=180,  # 6 months
        test_period=30,           # 1 month
        step=30                   # 1 month step
    )
    
    # Print walk-forward results
    print("\nWalk-Forward Analysis Results:")
    print(f"Average Test Return: {wf_results['walk_forward_metrics']['avg_test_return']*100:.2f}%")
    print(f"Test Return Std Dev: {wf_results['walk_forward_metrics']['std_test_return']*100:.2f}%")
    print(f"Train-Test Consistency: {wf_results['walk_forward_metrics']['consistency']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())