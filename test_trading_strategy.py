# tests/unit/test_trading_strategy.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from trading.strategy import TradingStrategy

class TestTradingStrategy:
    @pytest.fixture
    def strategy(self):
        return TradingStrategy(
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'close': np.random.normal(100, 10, 1000).cumsum(),
            'volume': np.random.randint(1000, 10000, 1000)
        })
    
    def test_rsi_calculation(self, strategy, sample_data):
        rsi = strategy._calculate_rsi(sample_data['close'])
        assert len(rsi) == len(sample_data)
        assert 0 <= rsi.min() <= 100
        assert 0 <= rsi.max() <= 100
        assert pd.notna(rsi).all()
    
    @patch('trading.strategy.TradingStrategy._calculate_rsi')
    def test_generate_signals(self, mock_rsi, strategy, sample_data):
        # Mock RSI values to test signal generation
        mock_rsi.return_value = pd.Series([75, 25, 75, 25], index=sample_data.index[:4])
        
        signals = strategy.generate_signals(sample_data)
        
        assert 'signal' in signals.columns
        assert set(signals['signal']).issubset([-1, 0, 1])
        assert len(signals) == len(sample_data)