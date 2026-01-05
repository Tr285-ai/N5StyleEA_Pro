import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd

from trading_strategy import TradingStrategy
from trading_bot import TradingBot

logger = logging.getLogger(__name__)

def _log_legacy_use(event: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs('logs', exist_ok=True)
        line = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': event,
            **payload,
        }
        with open(os.path.join('logs', 'deprecation.log'), 'a', encoding='utf-8') as f:
            f.write(str(line) + "\n")
    except Exception:
        # Telemetry is best-effort; never raise
        pass

async def run_legacy_strategy(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compatibility wrapper to route legacy strategy execution through the unified
    TradingBot orchestrator.

    strategy_config keys (best-effort mapping):
    - symbol: required trading symbol (e.g., 'BTC/USDT')
    - side: optional 'BUY' or 'SELL'; if omitted, AI strategy decides
    - quantity: optional notional fraction; we approximate using 10% balance
    - order_type: optional, currently MARKET-like behavior is supported
    - timeframe, exchange, ohlcv_limit: optional overrides
    """
    logger.warning("Using legacy compatibility wrapper; routing via TradingBot orchestrator. This path will be deprecated.")
    _log_legacy_use('legacy_wrapper_called', {'source': 'async', 'symbol': strategy_config.get('symbol')})
    symbol = strategy_config.get('symbol') or os.getenv('DEFAULT_SYMBOL')
    if not symbol:
        raise ValueError("strategy_config.symbol is required")

    timeframe = strategy_config.get('timeframe') or os.getenv('TIMEFRAME', '1m')
    exchange_name = strategy_config.get('exchange') or os.getenv('EXCHANGE_NAME', 'binance')
    ohlcv_limit = int(strategy_config.get('ohlcv_limit') or os.getenv('OHLCV_LIMIT', '100'))

    paper_mode = os.getenv('PAPER_TRADING', 'true').lower() in {'1', 'true', 'yes', 'y', 'on'}
    api_key = os.getenv('EXCHANGE_API_KEY')
    api_secret = os.getenv('EXCHANGE_API_SECRET')

    strategy = TradingStrategy(
        model_path=os.getenv('MODEL_PATH', 'models/ensemble_model'),
        input_shape=(30, 5),
        email_config=None,
        econ_calendar_api_key=os.getenv('ECONOMIC_CALENDAR_API_KEY'),
    )

    bot = TradingBot(
        strategy=strategy,
        symbols=[symbol],
        initial_balance=float(os.getenv('INITIAL_BALANCE', '10000')),
        exchange_name=exchange_name,
        timeframe=timeframe,
        ohlcv_limit=ohlcv_limit,
        paper=paper_mode,
        api_key=api_key,
        api_secret=api_secret,
    )

    # Fetch market data
    md: pd.DataFrame = await bot._get_market_data(symbol)

    # Build or derive a signal
    side = (strategy_config.get('side') or '').upper().strip()
    if side in {'BUY', 'SELL'}:
        price = float(md['close'].iloc[-1]) if (md is not None and len(md) and 'close' in md.columns) else strategy_config.get('price', 0.0)
        signal = {
            'symbol': symbol,
            'signal': side,
            'confidence': 0.75,
            'timestamp': datetime.utcnow().isoformat(),
            'price': price,
            'session': 'compat',
        }
    else:
        signal = await strategy.generate_signal(md, symbol)

    result: Dict[str, Any] = {'status': 'noop'}
    if signal.get('signal') != 'HOLD':
        await bot.execute_trade(symbol, signal)
        result = {
            'status': 'executed',
            'symbol': symbol,
            'side': signal.get('signal'),
            'price': signal.get('price'),
            'balance': bot.balance,
            'positions': bot.positions,
        }
    else:
        result = {'status': 'hold', 'symbol': symbol, 'reason': signal.get('reason')}

    return result


def run(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous convenience wrapper for environments not using asyncio."""
    logger.warning("Using legacy compatibility wrapper (sync); routing via TradingBot orchestrator. This path will be deprecated.")
    _log_legacy_use('legacy_wrapper_called', {'source': 'sync', 'symbol': strategy_config.get('symbol')})
    return asyncio.run(run_legacy_strategy(strategy_config))
