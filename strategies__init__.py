from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Type of trading signal."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()

@dataclass
class TradeSignal:
    """Represents a trading signal with entry, exit, and risk management parameters."""
    symbol: str
    signal: SignalType
    price: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    size: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, **kwargs):
        """Initialize the strategy with parameters."""
        self.params = kwargs
        self.name = self.__class__.__name__
        self.signals: List[TradeSignal] = []
        self.equity_curve: List[float] = []
        self.trades: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
    @abstractmethod
    async def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals based on the input data."""
        pass
        
    def set_parameters(self, **params) -> None:
        """Update strategy parameters."""
        self.params.update(params)
        self.logger.info(f"Updated parameters: {self.params}")
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.params.copy()
        
    async def backtest(
        self, 
        data: pd.DataFrame, 
        initial_balance: float = 10000.0,
        commission: float = 0.001
    ) -> Dict[str, Any]:
        """
        Run a backtest of the strategy.
        
        Args:
            data: DataFrame with price data
            initial_balance: Starting capital
            commission: Trading commission per trade
            
        Returns:
            Dictionary with backtest results
        """
        self.signals = await self.generate_signals(data)
        if not self.signals:
            return {"error": "No signals generated"}
            
        balance = initial_balance
        position = 0
        entry_price = 0
        self.trades = []
        self.equity_curve = [initial_balance]
        
        for signal in self.signals:
            if signal.signal == SignalType.BUY and position <= 0:
                # Close any existing position
                if position < 0:
                    pnl = (entry_price - signal.price) * abs(position) - commission
                    balance += pnl
                    
                # Open new long position
                position = (balance * 0.95) / signal.price  # Use 95% of balance
                entry_price = signal.price
                self.trades.append({
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'entry_time': signal.timestamp,
                    'size': position
                })
                
            elif signal.signal == SignalType.SELL and position >= 0:
                if position > 0:  # Close long position
                    pnl = (signal.price - entry_price) * position - commission
                    balance += pnl
                    
                # Open new short position
                position = -(balance * 0.95) / signal.price
                entry_price = signal.price
                self.trades.append({
                    'type': 'SHORT',
                    'entry_price': entry_price,
                    'entry_time': signal.timestamp,
                    'size': abs(position)
                })
                
            self.equity_curve.append(balance)
            
        # Calculate performance metrics
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = np.sqrt(252) * (returns.mean() / (returns.std() + 1e-9))
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': (balance / initial_balance - 1) * 100,
            'sharpe_ratio': float(sharpe),
            'max_drawdown': self._calculate_max_drawdown(),
            'num_trades': len(self.trades),
            'trades': self.trades
        }
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100

# Strategy Registry
_STRATEGY_REGISTRY = {
    'MovingAverageCrossover': None  # Will be set after class definition
}

def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to get a strategy by name.
    
    Args:
        name: Name of the strategy
        **kwargs: Strategy parameters
        
    Returns:
        An instance of the requested strategy
    """
    strategy_class = _STRATEGY_REGISTRY.get(name)
    if not strategy_class:
        available = list(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return strategy_class(**kwargs)

def list_strategies() -> List[str]:
    """Get a list of available strategy names."""
    return list(_STRATEGY_REGISTRY.keys())

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover strategy with RSI confirmation.
    Generates signals based on the crossing of two moving averages.
    """
    
    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        fast_window: int = 10,
        slow_window: int = 30,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        **kwargs
    ):
        """
        Initialize the strategy.
        
        Args:
            symbol: Trading pair symbol
            fast_window: Window for fast moving average
            slow_window: Window for slow moving average
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
        """
        super().__init__(**kwargs)
        self.symbol = symbol
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        # Update params for reference
        self.params.update({
            'symbol': symbol,
            'fast_window': fast_window,
            'slow_window': slow_window,
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold
        })
        
    async def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    async def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals based on moving average crossover."""
        if 'close' not in data.columns:
            raise ValueError("Input data must contain 'close' column")
            
        df = data.copy()
        signals = []
        
        # Calculate indicators
        df['fast_ma'] = df['close'].rolling(window=self.fast_window).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_window).mean()
        df['rsi'] = await self.calculate_rsi(df['close'])
        
        # Generate signals
        for i in range(1, len(df)):
            if pd.isna(df['fast_ma'].iloc[i]) or pd.isna(df['slow_ma'].iloc[i]):
                continue
                
            timestamp = df.index[i]
            price = df['close'].iloc[i]
            rsi = df['rsi'].iloc[i]
            
            prev_fast = df['fast_ma'].iloc[i-1]
            prev_slow = df['slow_ma'].iloc[i-1]
            curr_fast = df['fast_ma'].iloc[i]
            curr_slow = df['slow_ma'].iloc[i]
            
            # Bullish crossover with RSI confirmation
            if (prev_fast <= prev_slow and curr_fast > curr_slow and 
                rsi < self.rsi_overbought):
                signals.append(TradeSignal(
                    symbol=self.symbol,
                    signal=SignalType.BUY,
                    price=price,
                    timestamp=timestamp,
                    stop_loss=price * 0.98,  # 2% stop loss
                    take_profit=price * 1.03,  # 3% take profit
                    metadata={
                        'strategy': self.name,
                        'rsi': rsi,
                        'fast_ma': curr_fast,
                        'slow_ma': curr_slow
                    }
                ))
                
            # Bearish crossover with RSI confirmation
            elif (prev_fast >= prev_slow and curr_fast < curr_slow and 
                  rsi > self.rsi_oversold):
                signals.append(TradeSignal(
                    symbol=self.symbol,
                    signal=SignalType.SELL,
                    price=price,
                    timestamp=timestamp,
                    stop_loss=price * 1.02,  # 2% stop loss
                    take_profit=price * 0.97,  # 3% take profit
                    metadata={
                        'strategy': self.name,
                        'rsi': rsi,
                        'fast_ma': curr_fast,
                        'slow_ma': curr_slow
                    }
                ))
                
        return signals

# Register the strategy
_STRATEGY_REGISTRY['MovingAverageCrossover'] = MovingAverageCrossover

__all__ = [
    'BaseStrategy',
    'MovingAverageCrossover',
    'SignalType',
    'TradeSignal',
    'get_strategy',
    'list_strategies'
]