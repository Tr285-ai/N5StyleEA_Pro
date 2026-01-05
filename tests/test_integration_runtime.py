import os
import asyncio
import pytest

from trading_strategy import TradingStrategy
from trading_bot import TradingBot
from legacy_compat import run as legacy_run


@pytest.mark.asyncio
async def test_data_signal_trade_paper():
    os.environ['PAPER_TRADING'] = 'true'
    strategy = TradingStrategy(
        model_path=os.getenv('MODEL_PATH', 'models/ensemble_model'),
        input_shape=(30, 5),
    )
    bot = TradingBot(
        strategy=strategy,
        symbols=['BTC/USDT'],
        initial_balance=1000.0,
        exchange_name=os.getenv('EXCHANGE_NAME', 'binance'),
        timeframe='1m',
        ohlcv_limit=30,
        paper=True,
    )

    md = await bot._get_market_data('BTC/USDT')
    assert md is not None
    sig = await strategy.generate_signal(md, 'BTC/USDT')
    assert 'signal' in sig

    if sig.get('signal') != 'HOLD':
        await bot.execute_trade('BTC/USDT', sig)

    assert isinstance(bot.balance, float)


def test_legacy_compat_run_sync():
    os.environ['PAPER_TRADING'] = 'true'
    result = legacy_run({'symbol': 'BTC/USDT'})
    assert 'status' in result
    assert result.get('symbol') == 'BTC/USDT'
