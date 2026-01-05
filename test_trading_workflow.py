# tests/integration/test_trading_workflow.py
import pytest
from unittest.mock import patch, MagicMock
from trading.workflow import TradingWorkflow
from data_management import DataManager

class TestTradingWorkflow:
    @pytest.fixture
    def workflow(self):
        return TradingWorkflow(
            strategy_config={'rsi_period': 14},
            data_source='binance',
            symbols=['BTC/USDT', 'ETH/USDT']
        )
    
    @patch('trading.workflow.DataManager')
    @patch('trading.workflow.ExchangeInterface')
    async def test_execute_trades(self, mock_exchange, mock_dm, workflow):
        # Setup mocks
        mock_dm.return_value.load_historical_data.return_value = pd.DataFrame()
        mock_exchange.return_value.get_balance.return_value = {'USDT': 10000}
        
        # Execute workflow
        await workflow.run()
        
        # Verify interactions
        mock_exchange.return_value.place_order.assert_called()
        mock_dm.return_value.save_trade.assert_called()

    @pytest.mark.parametrize("market_condition", ["bullish", "bearish", "volatile"])
    async def test_market_conditions(self, workflow, market_condition):
        # Test different market conditions
        with patch.object(workflow.strategy, 'generate_signals') as mock_signals:
            if market_condition == "bullish":
                mock_signals.return_value = pd.Series([1] * 100)
            elif market_condition == "bearish":
                mock_signals.return_value = pd.Series([-1] * 100)
            else:
                mock_signals.return_value = pd.Series(np.random.choice([-1, 0, 1], 100))
            
            await workflow.run()
            # Add assertions based on expected behavior