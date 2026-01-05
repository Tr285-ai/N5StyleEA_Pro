import pytest
from datetime import datetime
from advanced_metrics import Trade, AdvancedMetrics

def test_advanced_metrics_calculation():
    # Create sample trades
    trades = [
        Trade(
            entry_time=datetime(2023, 1, 1),
            exit_time=datetime(2023, 1, 2),
            entry_price=100.0,
            exit_price=110.0,
            size=1.0,
            side="long",
            pnl=10.0,
            pnl_pct=0.10,
        ),
        Trade(
            entry_time=datetime(2023, 1, 3),
            exit_time=datetime(2023, 1, 4),
            entry_price=110.0,
            exit_price=99.0,
            size=1.0,
            side="long",
            pnl=-11.0,
            pnl_pct=-0.10,
        ),
    ]
    
    # Calculate metrics
    metrics = AdvancedMetrics().calculate_all_metrics(trades)
    
    # Assertions
    assert metrics["total_trades"] == 2
    assert metrics["win_rate"] == 50.0
    assert metrics["profit_factor"] > 0
    assert metrics["max_drawdown"] < 0

def test_trade_analysis_report():
    # Create sample trades
    trades = [
        Trade(
            entry_time=datetime(2023, 1, 1),
            exit_time=datetime(2023, 1, 2),
            entry_price=100.0,
            exit_price=110.0,
            size=1.0,
            side="long",
            pnl=10.0,
            pnl_pct=0.10,
        )
    ]
    
    # Generate report
    from advanced_metrics import create_trade_analysis_report
    report = create_trade_analysis_report(trades)
    
    # Basic assertions
    assert "TRADE ANALYSIS REPORT" in report
    assert "Initial Capital" in report
    assert "Final Equity" in report
    assert "Performance Metrics" in report