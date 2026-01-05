import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from moving_average_crossover import MovingAverageCrossover

@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    date_rng = pd.date_range(start='2023-01-01', end='2023-02-01', freq='D')
    prices = np.cumsum(np.random.randn(len(date_rng)) * 0.01 + 0.001) + 100
    return pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(100, 1000, size=len(date_rng))
    }, index=date_rng)

@pytest.fixture
def strategy():
    """Create a strategy instance for testing."""
    return MovingAverageCrossover({
        'fast_ma_period': 5,
        'slow_ma_period': 20,
        'stop_loss_pct': 1.0,
        'take_profit_pct': 2.0
    })

@pytest.mark.asyncio
async def test_initialization(strategy):
    """Test strategy initialization."""
    await strategy.initialize()
    assert strategy.initialized is True

@pytest.mark.asyncio
async def test_initialization_invalid_periods():
    """Test initialization with invalid period values."""
    with pytest.raises(ValueError):
        strategy = MovingAverageCrossover({
            'fast_ma_period': 20,
            'slow_ma_period': 10  # Fast MA > Slow MA should raise error
        })
        await strategy.initialize()

@pytest.mark.asyncio
async def test_analyze(strategy, sample_data):
    """Test signal generation."""
    await strategy.initialize()
    result = await strategy.analyze(sample_data)
    
    # Check if the result has the expected structure
    assert 'signals' in result
    assert 'indicators' in result
    assert 'metadata' in result
    
    # Check if indicators were calculated
    assert 'fast_ma' in result['indicators']
    assert 'slow_ma' in result['indicators']
    
    # Check if signals are generated (may be empty if no crossovers)
    assert isinstance(result['signals'], list)

@pytest.mark.asyncio
async def test_execute(strategy):
    """Test order execution."""
    await strategy.initialize()
    
    test_signals = [{
        'type': 'LONG',
        'price': 100.0,
        'stop_loss': 99.0,
        'take_profit': 102.0,
        'symbol': 'BTC/USDT'
    }]
    
    result = await strategy.execute(test_signals)
    
    # Check if the result has the expected structure
    assert 'status' in result
    assert 'executed_orders' in result
    assert len(result['executed_orders']) == len(test_signals)
    assert result['executed_orders'][0]['status'] == 'executed'

@pytest.mark.asyncio
async def test_empty_signals(strategy):
    """Test execution with empty signals."""
    await strategy.initialize()
    result = await strategy.execute([])
    assert result['status'] == 'no_signals'
    assert len(result['executed_orders']) == 0

def test_string_representation(strategy):
    """Test string representation of the strategy."""
    assert "MovingAverageCrossover" in str(strategy)
    assert "fast_ma=5" in str(strategy)
    assert "slow_ma=20" in str(strategy)

@pytest.mark.asyncio
async def test_analyze_insufficient_data(strategy):
    """Test analysis with insufficient data."""
    await strategy.initialize()
    
    # Create a DataFrame with fewer data points than the slow MA period
    small_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100, 101, 102],
        'volume': [1000, 1200, 1500]
    })
    
    result = await strategy.analyze(small_data)
    assert len(result['signals']) == 0  # No signals should be generated

@pytest.mark.asyncio
async def test_analyze_with_nan_values(strategy, sample_data):
    """Test analysis with NaN values in the data."""
    await strategy.initialize()
    
    # Introduce NaN values
    sample_data_with_nan = sample_data.copy()
    sample_data_with_nan.iloc[10:15] = np.nan
    
    result = await strategy.analyze(sample_data_with_nan)
    # Should handle NaNs gracefully
    assert isinstance(result, dict)