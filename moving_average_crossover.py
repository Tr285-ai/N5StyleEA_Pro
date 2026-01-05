import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from .base import BaseStrategy

logger = logging.getLogger(__name__)

@dataclass
class MovingAverageCrossoverConfig:
    """Configuration for Moving Average Crossover strategy."""
    fast_ma_period: int = 10
    slow_ma_period: int = 30
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 2.0
    max_position_size: float = 0.1  # 10% of portfolio

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover strategy that generates signals based on the
    crossover of fast and slow moving averages.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("MovingAverageCrossover", config or {})
        self.config = MovingAverageCrossoverConfig(**self.config)
        self.initialized = False
        
    async def _initialize_resources(self) -> None:
        """Initialize any resources needed by the strategy."""
        # Validate configuration
        if self.config.fast_ma_period >= self.config.slow_ma_period:
            raise ValueError("Fast MA period must be less than slow MA period")
            
        if self.config.fast_ma_period < 1 or self.config.slow_ma_period < 1:
            raise ValueError("MA periods must be positive integers")
            
        self.initialized = True
        logger.info(f"Initialized MovingAverageCrossover strategy with config: {self.config}")
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages and signals.
        
        Args:
            df: DataFrame with 'close' price column
            
        Returns:
            DataFrame with added MA and signal columns
        """
        df = df.copy()
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.config.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.config.slow_ma_period).mean()
        
        # Generate signals
        df['signal'] = 0  # 0 = no position, 1 = long, -1 = short
        
        # Long signal: fast MA crosses above slow MA
        df.loc[(df['fast_ma'] > df['slow_ma']) & 
              (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)), 'signal'] = 1
              
        # Short signal: fast MA crosses below slow MA
        df.loc[(df['fast_ma'] < df['slow_ma']) & 
              (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)), 'signal'] = -1
        
        return df
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data: DataFrame with market data (must include 'close' column)
            
        Returns:
            Dictionary containing analysis results and signals
        """
        if not self.initialized:
            await self.initialize()
            
        if data.empty:
            logger.warning("Empty data received for analysis")
            return {'signals': [], 'indicators': {}}
        
        try:
            # Calculate indicators
            df = self._calculate_moving_averages(data)
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Prepare signals
            signals = []
            if latest_signal != 0:
                signals.append({
                    'type': 'LONG' if latest_signal > 0 else 'SHORT',
                    'price': df['close'].iloc[-1],
                    'timestamp': df.index[-1],
                    'stop_loss': df['close'].iloc[-1] * 
                               (1 - self.config.stop_loss_pct/100 if latest_signal > 0 
                                else 1 + self.config.stop_loss_pct/100),
                    'take_profit': df['close'].iloc[-1] * 
                                 (1 + self.config.take_profit_pct/100 if latest_signal > 0 
                                  else 1 - self.config.take_profit_pct/100)
                })
            
            # Prepare indicators for visualization
            indicators = {
                'fast_ma': df['fast_ma'].to_dict(),
                'slow_ma': df['slow_ma'].to_dict()
            }
            
            return {
                'signals': signals,
                'indicators': indicators,
                'metadata': {
                    'fast_ma_period': self.config.fast_ma_period,
                    'slow_ma_period': self.config.slow_ma_period
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}", exc_info=True)
            raise
            
    async def execute(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute trades based on signals.
        
        Args:
            signals: List of trading signals
            
        Returns:
            Dictionary with execution results
        """
        if not signals:
            return {'status': 'no_signals', 'executed_orders': []}
            
        results = []
        for signal in signals:
            try:
                # In a real implementation, this would place actual orders
                order = {
                    'symbol': signal.get('symbol', 'UNKNOWN'),
                    'side': signal['type'],
                    'price': signal['price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'size': self.config.max_position_size,
                    'status': 'executed',
                    'order_id': f"order_{pd.Timestamp.now().value}",
                    'timestamp': pd.Timestamp.now()
                }
                results.append(order)
                logger.info(f"Executed order: {order}")
                
            except Exception as e:
                logger.error(f"Error executing order: {str(e)}", exc_info=True)
                results.append({
                    'status': 'error',
                    'error': str(e),
                    'signal': signal
                })
        
        return {
            'status': 'completed',
            'executed_orders': results
        }
        
    def __str__(self) -> str:
        return (f"MovingAverageCrossover(fast_ma={self.config.fast_ma_period}, "
               f"slow_ma={self.config.slow_ma_period})")