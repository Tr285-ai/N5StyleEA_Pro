# tests/integration/test_integration.py
import pytest
import pandas as pd
import numpy as np
from ml_core_v15_2 import TradingSystem, SignalEngine, TradingStrategy
from datetime import datetime, timedelta

class TestIntegration:
    @pytest.fixture
    def trading_system(self):
        return TradingSystem(
            config_path='config/test_config.yaml',
            data_feed=MockDataFeed(),
            broker=MockBroker()
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_trading(self, trading_system, sample_ohlcv_data):
        """Test complete trading cycle."""
        # Initialize system
        await trading_system.initialize()
        
        # Process market data
        for idx, row in sample_ohlcv_data.iterrows():
            await trading_system.process_market_data('TEST', row.to_dict())
            
            # Check if orders are being created
            if len(trading_system.broker.orders) > 0:
                # Process open orders
                for order in trading_system.broker.orders:
                    if order['status'] == 'open':
                        await trading_system.broker.update_order_status(
                            order['id'],
                            'filled',
                            order['price']
                        )
        
        # Verify final state
        assert trading_system.strategy.current_balance > 0
        assert len(trading_system.broker.trades) > 0