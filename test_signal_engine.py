# c:\N5StyleEA_Pro v15_3\test_signal_engine.py
import asyncio
import logging
import pytest
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

from .signal_engine import SignalEngine
from .models import Signal, SignalType, Timeframe

logger = logging.getLogger(__name__)

class TestSignalEngine:
    """Test cases for the SignalEngine class."""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        return pd.DataFrame({
            "timestamp": dates,
            "open": close_prices - np.random.rand(100) * 0.5,
            "high": close_prices + np.random.rand(100) * 0.5,
            "low": close_prices - np.random.rand(100) * 0.5,
            "close": close_prices,
            "volume": np.random.randint(100, 1000, size=100)
        }).set_index("timestamp")
        
    @pytest.fixture
    def signal_engine(self) -> SignalEngine:
        """Create a SignalEngine instance for testing."""
        return SignalEngine()
        
    @pytest.mark.asyncio
    async def test_generate_signals(self, signal_engine: SignalEngine, sample_data: pd.DataFrame):
        """Test signal generation."""
        # Test with default parameters
        signals = await signal_engine.generate_signals(
            data=sample_data,
            symbol="TEST",
            timeframe=Timeframe.H1
        )
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.symbol == "TEST"
            assert signal.timeframe == Timeframe.H1
            assert signal.type in [SignalType.BUY, SignalType.SELL, SignalType.NEUTRAL]
            assert signal.strength >= 0 and signal.strength <= 1
            
    @pytest.mark.asyncio
    async def test_combine_signals(self, signal_engine: SignalEngine):
        """Test signal combination logic."""
        # Create test signals
        signals = [
            Signal(
                symbol="TEST",
                type=SignalType.BUY,
                strength=0.8,
                timestamp=datetime.now(),
                timeframe=Timeframe.H1,
                source="test"
            ),
            Signal(
                symbol="TEST",
                type=SignalType.SELL,
                strength=0.6,
                timestamp=datetime.now(),
                timeframe=Timeframe.H1,
                source="test"
            )
        ]
        
        # Test combining signals
        combined = await signal_engine.combine_signals(signals)
        
        assert isinstance(combined, Signal)
        assert combined.type in [SignalType.BUY, SignalType.SELL, SignalType.NEUTRAL]
        
    @pytest.mark.asyncio
    async def test_signal_strength_calculation(self, signal_engine: SignalEngine, sample_data: pd.DataFrame):
        """Test signal strength calculation."""
        # Test with sample data
        signals = await signal_engine.generate_signals(
            data=sample_data,
            symbol="TEST",
            timeframe=Timeframe.H1
        )
        
        # Verify signal strengths are within expected range
        for signal in signals:
            assert 0 <= signal.strength <= 1
            
    @pytest.mark.asyncio
    async def test_empty_data(self, signal_engine: SignalEngine):
        """Test with empty data."""
        empty_data = pd.DataFrame()
        signals = await signal_engine.generate_signals(
            data=empty_data,
            symbol="TEST",
            timeframe=Timeframe.H1
        )
        
        assert signals == []
        
    @pytest.mark.asyncio
    async def test_invalid_data(self, signal_engine: SignalEngine):
        """Test with invalid data."""
        invalid_data = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "invalid": [4, 5, 6]
        })
        
        with pytest.raises(ValueError):
            await signal_engine.generate_signals(
                data=invalid_data,
                symbol="TEST",
                timeframe=Timeframe.H1
            )