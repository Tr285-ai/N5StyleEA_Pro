import asyncio
import logging
from typing import Dict, List, Optional
import os
import random
import json
import numpy as np
import pandas as pd
from datetime import datetime
from trading_strategy import TradingStrategy

from executor import TradeExecutor
from risk.advanced_controls import Portfolio, Position, PositionLimit, LossLimit, ConcentrationLimit

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(
        self,
        strategy: TradingStrategy,
        symbols: List[str],
        initial_balance: float = 10000.0,
        exchange_name: str = "binance",
        timeframe: str = "1m",
        ohlcv_limit: int = 100,
        paper: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):

        self.strategy = strategy
        self.symbols = [s.strip() for s in symbols if s and s.strip()]
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.trade_history = []
        self.running = False

        self.exchange_name = exchange_name
        self.timeframe = timeframe
        self.ohlcv_limit = ohlcv_limit

        if api_key is None:
            api_key = os.getenv('EXCHANGE_API_KEY')
        if api_secret is None:
            api_secret = os.getenv('EXCHANGE_API_SECRET')

        self.executor = TradeExecutor(
            {
                'broker': {
                    'demo': bool(paper),
                    'exchange': exchange_name,
                    'api_key': api_key,
                    'api_secret': api_secret,
                }
            }
        )

        self.portfolio = Portfolio()
        self.portfolio.cash = float(initial_balance)
        max_position_size = {s: float(os.getenv('MAX_POSITION_SIZE', '1000000')) for s in self.symbols}
        max_notional_value = {s: float(os.getenv('MAX_POSITION_NOTIONAL', str(initial_balance))) for s in self.symbols}
        self.portfolio.add_risk_limit(PositionLimit(max_position_size=max_position_size, max_notional_value=max_notional_value))
        self.portfolio.add_risk_limit(LossLimit(
            max_daily_loss_pct=float(os.getenv('MAX_DAILY_LOSS_PCT', '0.05')),
            max_daily_loss_abs=float(os.getenv('MAX_DAILY_LOSS_ABS', str(initial_balance * 0.05))),
        ))
        self.portfolio.add_risk_limit(ConcentrationLimit(
            max_single_position_pct=float(os.getenv('MAX_SINGLE_POSITION_PCT', '0.25')),
        ))

        self._exchange = None
        self._state_file = os.getenv('BOT_STATE_FILE', 'state/trading_bot_state.json')
        try:
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        except Exception:
            pass

    async def run(self) -> None:
        """Main trading loop"""
        self.running = True
        logger.info(f"Starting trading bot with balance: ${self.balance:.2f}")
        
        try:
            await self.executor.initialize()
            await self._load_state()
        if str(os.getenv('USE_EVENT_DRIVEN', '0')).lower() not in ('0', 'false', 'no'):
            await self._run_event_driven()
            return
            while self.running:
                for symbol in self.symbols:
                    try:
                        # Get market data (implement this based on your data source)
                        market_data = await self._get_market_data(symbol)
                        
                        # Generate trading signal
                        signal = await self.strategy.generate_signal(market_data, symbol)
                        
                        # Execute trade based on signal
                        if signal['signal'] != 'HOLD':
                            await self.execute_trade(symbol, signal)
                            
                        # Update RL agent with new data
                        await self._update_rl_agent(market_data)
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                        
                # Sleep for a while before next iteration
                await asyncio.sleep(60)  # 1 minute
                
        except asyncio.CancelledError:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {str(e)}")
            raise
        finally:
            self.running = False
            try:
                await self._save_state()
            except Exception:
                pass

    async def execute_trade(self, symbol: str, signal: Dict) -> None:
        """Execute a trade based on the signal"""
        try:
            price = signal.get('price', 1.0)  # Get current price from signal or market data
            amount = self.balance * 0.1 / price  # Example: 10% of balance

            side = signal['signal']
            signed_qty = float(amount) if side == 'BUY' else -float(amount)

            self.portfolio.cash = float(self.balance)
            self.portfolio.update_market_data({symbol: float(price)})

            order = {
                'symbol': symbol,
                'side': side,
                'quantity': signed_qty,
                'price': float(price),
                'timestamp': datetime.utcnow(),
            }

            passed, results = self.portfolio.check_risk(order)
            if not passed:
                msg = "; ".join(str(r) for r in results if not r.passed)
                logger.warning(f"Risk check blocked order for {symbol}: {msg}")
                return

            if side == 'SELL' and self.positions.get(symbol, 0.0) < amount:
                return

            execution = await self.executor.execute_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                price=float(price),
                quantity=float(amount),
            )
            if execution.get('status') != 'filled':
                return

            if side == 'BUY':
                self.positions[symbol] = self.positions.get(symbol, 0.0) + amount
                self.balance -= amount * price
                logger.info(f"Bought {amount} of {symbol} at {price}")
            elif side == 'SELL':
                self.positions[symbol] = self.positions.get(symbol, 0.0) - amount
                self.balance += amount * price
                logger.info(f"Sold {amount} of {symbol} at {price}")

            pos = self.portfolio.get_position(symbol)
            if pos is None:
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=float(self.positions.get(symbol, 0.0)),
                    avg_price=float(price),
                    current_price=float(price),
                    timestamp=datetime.utcnow(),
                )
            else:
                pos.quantity = float(self.positions.get(symbol, 0.0))
                pos.update_price(float(price))

            # Record trade
            self.trade_history.append({
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'action': signal['signal'],
                'price': price,
                'amount': amount,
                'balance': self.balance
            })
            self.portfolio.trade_history.append({'timestamp': datetime.utcnow(), 'symbol': symbol, 'pnl': 0.0})
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise

    async def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for a symbol"""
        try:
            market_data = await self._fetch_ohlcv_ccxt(symbol)
            if market_data is not None and not market_data.empty:
                return market_data
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")

        return pd.DataFrame({
            'open': np.random.random(self.ohlcv_limit),
            'high': np.random.random(self.ohlcv_limit),
            'low': np.random.random(self.ohlcv_limit),
            'close': np.random.random(self.ohlcv_limit),
            'volume': np.random.random(self.ohlcv_limit)
        })

    async def _fetch_ohlcv_ccxt(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import ccxt
        except Exception:
            return None

        if self._exchange is None:
            ex_class = getattr(ccxt, (self.exchange_name or "").lower(), None)
            if ex_class is None:
                return None
            self._exchange = ex_class({'enableRateLimit': True, 'timeout': int(os.getenv('CCXT_TIMEOUT_MS', '10000'))})
        retries = int(os.getenv('CCXT_MAX_RETRIES', '3'))
        backoff_ms = int(os.getenv('CCXT_RETRY_BACKOFF_MS', '1000'))
        ohlcv = None
        last_err = None
        for attempt in range(max(1, retries)):
            try:
                ohlcv = await asyncio.to_thread(
                    self._exchange.fetch_ohlcv,
                    symbol,
                    self.timeframe,
                    None,
                    int(self.ohlcv_limit),
                )
                if ohlcv:
                    break
            except Exception as e:
                last_err = e
                delay = (backoff_ms * (attempt + 1)) / 1000.0
                delay += random.uniform(0, 0.25)
                await asyncio.sleep(delay)
        if not ohlcv:
            if last_err:
                logger.warning(f"fetch_ohlcv failed after retries for {symbol}: {last_err}")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    async def _run_event_driven(self) -> None:
        self.running = True
        try:
            tasks = [asyncio.create_task(self._symbol_loop(sym)) for sym in self.symbols]
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False

    async def _symbol_loop(self, symbol: str) -> None:
        while self.running:
            try:
                market_data = await self._get_market_data(symbol)
                signal = await self.strategy.generate_signal(market_data, symbol)
                if signal.get('signal') and signal['signal'] != 'HOLD':
                    await self.execute_trade(symbol, signal)
                await self._update_rl_agent(market_data)
            except Exception as e:
                logger.error(f'Error in symbol loop {symbol}: {e}')
            # Sleep until the next candle boundary
            try:
                delay = self._seconds_until_next_candle(self.timeframe)
            except Exception:
                delay = 60.0
            await asyncio.sleep(max(0.5, float(delay)))

    def _seconds_until_next_candle(self, timeframe: str) -> float:
        now = datetime.utcnow()
        tf = (timeframe or '1m').strip().lower()
        # Parse timeframe
        num = 1
        unit = 'm'
        try:
            if tf[-1] in ('s','m','h','d'):
                unit = tf[-1]
                num = int(tf[:-1]) if tf[:-1].isdigit() else 1
            else:
                num = int(tf)
                unit = 'm'
        except Exception:
            num, unit = 1, 'm'
        seconds = num * (1 if unit=='s' else 60 if unit=='m' else 3600 if unit=='h' else 86400)
        # Compute next boundary
        ts = int(now.timestamp())
        next_boundary = ((ts // seconds) + 1) * seconds
        return float(next_boundary - ts)


    async def _update_rl_agent(self, market_data: pd.DataFrame) -> None:
        """Update RL agent with new market data"""
        try:
            # Implement RL agent update logic here
            pass
        except Exception as e:
            logger.error(f"Error updating RL agent: {str(e)}")

    async def _load_state(self) -> None:
        try:
            if not self._state_file or not os.path.exists(self._state_file):
                return
            with open(self._state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.balance = float(data.get('balance', self.balance))
                positions = data.get('positions') or {}
                if isinstance(positions, dict):
                    self.positions = {str(k): float(v) for k, v in positions.items()}
        except Exception as e:
            logger.warning(f"Failed to load bot state: {e}")

    async def _save_state(self) -> None:
        try:
            state = {
                'timestamp': datetime.utcnow().isoformat(),
                'balance': float(self.balance),
                'positions': {k: float(v) for k, v in self.positions.items()},
            }
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"Failed to save bot state: {e}")
