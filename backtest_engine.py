import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
from ..strategies.base import BaseStrategy
from ..data.providers import DataProvider

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Represents the results of a backtest."""
    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    config: Dict[str, Any]
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown_pct: float
    num_trades: int
    win_loss_ratio: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

class BacktestEngine:
    """Backtesting engine for evaluating trading strategies."""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: DataProvider,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d"
    ):
        """
        Initialize the backtest engine.
        
        Args:
            strategy: The trading strategy to backtest
            data_provider: Data provider for market data
            initial_balance: Starting balance in quote currency
            commission: Trading commission as a fraction (e.g., 0.001 for 0.1%)
            slippage: Slippage as a fraction of price
            start_date: Backtest start date (inclusive)
            end_date: Backtest end date (exclusive)
            symbol: Trading pair symbol
            timeframe: Data timeframe
        """
        self.strategy = strategy
        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Convert string dates to datetime if needed
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else datetime.utcnow()
        
        # Internal state
        self._data = None
        self._results = None
        self._trades = []
        self._equity_curve = []
        self._current_balance = initial_balance
        self._current_position = 0.0
        self._current_price = 0.0
        self._current_time = None
        self._trade_id = 0
        
        # Performance metrics
        self._metrics = {}
        
    async def run(self) -> BacktestResult:
        """Run the backtest."""
        try:
            logger.info(f"Starting backtest for {self.strategy.__class__.__name__}")
            
            # Load historical data
            await self._load_data()
            
            if self._data.empty:
                raise ValueError("No data available for the specified date range")
                
            # Initialize strategy
            await self.strategy.initialize()
            
            # Run backtest
            await self._execute_backtest()
            
            # Calculate performance metrics
            await self._calculate_metrics()
            
            # Generate results
            results = self._generate_results()
            
            logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            raise
            
    async def _load_data(self) -> None:
        """Load historical data for the backtest."""
        logger.info(f"Loading data for {self.symbol} ({self.timeframe}) from {self.start_date} to {self.end_date}")
        
        # Load data using the data provider
        self._data = await self.data_provider.get_historical_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if self._data is None or self._data.empty:
            raise ValueError("No data returned from data provider")
            
        logger.info(f"Loaded {len(self._data)} data points")
        
    async def _execute_backtest(self) -> None:
        """Execute the backtest by iterating through the data."""
        logger.info("Executing backtest...")
        
        # Initialize variables
        self._current_balance = self.initial_balance
        self._current_position = 0.0
        self._trades = []
        self._equity_curve = []
        
        # Iterate through each bar
        for i, (timestamp, row) in enumerate(self._data.iterrows()):
            self._current_time = timestamp
            self._current_price = row['close']
            
            # Update strategy with current market data
            market_data = row.to_dict()
            market_data['timestamp'] = timestamp
            
            # Generate signals
            signals = await self.strategy.analyze(market_data)
            
            # Execute trades based on signals
            if signals and 'signals' in signals:
                await self._process_signals(signals['signals'])
                
            # Update equity curve
            position_value = self._current_position * self._current_price
            self._equity_curve.append({
                'timestamp': timestamp,
                'balance': self._current_balance,
                'position': self._current_position,
                'position_value': position_value,
                'equity': self._current_balance + position_value,
                'price': self._current_price
            })
            
    async def _process_signals(self, signals: List[Dict[str, Any]]) -> None:
        """Process trading signals and execute orders."""
        for signal in signals:
            try:
                if signal.get('action') == 'buy' and self._current_position <= 0:
                    # Calculate position size based on available balance
                    size = (self._current_balance * 0.99) / self._current_price  # 99% of balance
                    
                    # Apply slippage
                    fill_price = self._current_price * (1 + self.slippage)
                    
                    # Calculate commission
                    commission = size * fill_price * self.commission
                    
                    # Update balance and position
                    cost = (size * fill_price) + commission
                    if cost > self._current_balance:
                        logger.warning("Insufficient balance for buy order")
                        continue
                        
                    self._current_balance -= cost
                    self._current_position += size
                    
                    # Record trade
                    self._record_trade(
                        action='buy',
                        size=size,
                        price=fill_price,
                        commission=commission
                    )
                    
                elif signal.get('action') == 'sell' and self._current_position > 0:
                    # Apply slippage
                    fill_price = self._current_price * (1 - self.slippage)
                    
                    # Calculate commission
                    commission = self._current_position * fill_price * self.commission
                    
                    # Update balance and position
                    proceeds = (self._current_position * fill_price) - commission
                    self._current_balance += proceeds
                    
                    # Record trade
                    self._record_trade(
                        action='sell',
                        size=self._current_position,
                        price=fill_price,
                        commission=commission
                    )
                    
                    self._current_position = 0.0
                    
            except Exception as e:
                logger.error(f"Error processing signal: {e}", exc_info=True)
                
    def _record_trade(
        self,
        action: str,
        size: float,
        price: float,
        commission: float
    ) -> None:
        """Record a trade."""
        self._trade_id += 1
        trade = {
            'id': self._trade_id,
            'timestamp': self._current_time,
            'action': action,
            'size': size,
            'price': price,
            'value': size * price,
            'commission': commission,
            'balance': self._current_balance,
            'position': self._current_position
        }
        self._trades.append(trade)
        logger.debug(f"Trade executed: {trade}")
        
    async def _calculate_metrics(self) -> None:
        """Calculate performance metrics."""
        if not self._trades:
            logger.warning("No trades were executed during backtest")
            return
            
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self._trades)
        trades_df['pnl'] = 0.0
        trades_df['return_pct'] = 0.0
        
        # Calculate P&L for each trade
        for i in range(1, len(trades_df), 2):
            if i < len(trades_df):
                buy_trade = trades_df.iloc[i-1]
                sell_trade = trades_df.iloc[i]
                
                if buy_trade['action'] == 'buy' and sell_trade['action'] == 'sell':
                    pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['size']
                    pnl -= buy_trade['commission'] + sell_trade['commission']
                    return_pct = (pnl / (buy_trade['price'] * buy_trade['size'])) * 100
                    
                    trades_df.at[i-1, 'pnl'] = pnl
                    trades_df.at[i, 'pnl'] = pnl
                    trades_df.at[i-1, 'return_pct'] = return_pct
                    trades_df.at[i, 'return_pct'] = return_pct
        
        # Calculate metrics
        total_return = (self._current_balance / self.initial_balance - 1) * 100
        num_trades = len(trades_df) // 2
        winning_trades = (trades_df['pnl'] > 0).sum() // 2
        losing_trades = (trades_df['pnl'] < 0).sum() // 2
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
        
        profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = profit / loss if loss != 0 else float('inf')
        
        # Calculate equity curve
        equity_curve = pd.DataFrame(self._equity_curve)
        equity_curve.set_index('timestamp', inplace=True)
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak']
        max_drawdown_pct = equity_curve['drawdown'].min() * 100
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        annualized_return = (equity_curve['returns'].mean() * 252 * 100)  # 252 trading days
        volatility = equity_curve['returns'].std() * np.sqrt(252) * 100  # Annualized
        
        # Calculate Sharpe ratio (risk-free rate assumed to be 0)
        sharpe_ratio = (equity_curve['returns'].mean() / equity_curve['returns'].std()) * np.sqrt(252)
        
        # Calculate Sortino ratio
        downside_returns = equity_curve[equity_curve['returns'] < 0]['returns']
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (equity_curve['returns'].mean() * 252 / downside_std) if downside_std != 0 else 0
        
        # Store metrics
        self._metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade': trades_df['pnl'].mean(),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'max_consecutive_wins': self._calculate_max_consecutive(trades_df, 'win'),
            'max_consecutive_losses': self._calculate_max_consecutive(trades_df, 'loss')
        }
        
    def _calculate_max_consecutive(self, trades_df: pd.DataFrame, trade_type: str) -> int:
        """Calculate maximum consecutive wins or losses."""
        if trade_type not in ['win', 'loss']:
            return 0
            
        is_win = trades_df['pnl'] > 0
        if trade_type == 'loss':
            is_win = ~is_win
            
        # Find consecutive sequences
        consecutive = is_win.astype(int)
        groups = consecutive.diff().ne(0).cumsum()
        counts = consecutive.groupby(groups).cumsum()
        
        return counts.max() if not counts.empty else 0
        
    def _generate_results(self) -> BacktestResult:
        """Generate backtest results."""
        # Create trades DataFrame
        trades_df = pd.DataFrame(self._trades)
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self._equity_curve)
        equity_curve = equity_df.set_index('timestamp')['equity']
        
        # Create positions DataFrame
        positions_df = equity_df[['timestamp', 'position']].copy()
        positions_df['price'] = equity_df['price']
        positions_df['value'] = positions_df['position'] * positions_df['price']
        
        return BacktestResult(
            returns=equity_curve.pct_change(),
            equity_curve=equity_curve,
            positions=positions_df,
            trades=trades_df,
            metrics=self._metrics,
            config={
                'strategy': self.strategy.__class__.__name__,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'initial_balance': self.initial_balance,
                'commission': self.commission,
                'slippage': self.slippage
            },
            strategy_name=self.strategy.__class__.__name__,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_balance=self.initial_balance,
            final_balance=self._current_balance,
            max_drawdown=self._metrics.get('max_drawdown_pct', 0),
            sharpe_ratio=self._metrics.get('sharpe_ratio', 0),
            sortino_ratio=self._metrics.get('sortino_ratio', 0),
            win_rate=self._metrics.get('win_rate', 0),
            profit_factor=self._metrics.get('profit_factor', 0),
            total_return=self._metrics.get('total_return', 0),
            annualized_return=self._metrics.get('annualized_return', 0),
            volatility=self._metrics.get('volatility', 0),
            max_drawdown_pct=self._metrics.get('max_drawdown_pct', 0),
            num_trades=self._metrics.get('num_trades', 0),
            win_loss_ratio=(self._metrics.get('winning_trades', 0) / 
                          self._metrics.get('losing_trades', 1)) if self._metrics.get('losing_trades', 0) > 0 else 0,
            avg_trade=self._metrics.get('avg_trade', 0),
            avg_win=self._metrics.get('avg_win', 0),
            avg_loss=self._metrics.get('avg_loss', 0),
            largest_win=self._metrics.get('largest_win', 0),
            largest_loss=self._metrics.get('largest_loss', 0),
            max_consecutive_wins=self._metrics.get('max_consecutive_wins', 0),
            max_consecutive_losses=self._metrics.get('max_consecutive_losses', 0)
        )
        
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save backtest results to a file."""
        if not self._results:
            raise ValueError("No results to save. Run the backtest first.")
            
        try:
            results = {
                'config': {
                    'strategy': self.strategy.__class__.__name__,
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'start_date': self.start_date.isoformat() if self.start_date else None,
                    'end_date': self.end_date.isoformat(),
                    'initial_balance': self.initial_balance,
                    'commission': self.commission,
                    'slippage': self.slippage
                },
                'metrics': self._metrics,
                'equity_curve': [{
                    'timestamp': ts.isoformat(),
                    'equity': eq
                } for ts, eq in self._results.equity_curve.items()],
                'trades': [{
                    'id': t['id'],
                    'timestamp': t['timestamp'].isoformat(),
                    'action': t['action'],
                    'size': t['size'],
                    'price': t['price'],
                    'value': t['value'],
                    'commission': t['commission'],
                    'balance': t['balance'],
                    'position': t['position']
                } for t in self._trades]
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved backtest results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise