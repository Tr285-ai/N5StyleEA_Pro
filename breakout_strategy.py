import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from .base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy that identifies and trades breakouts from key levels.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Breakout strategy.
        
        Config options:
            atr_period: ATR period for volatility measurement (default: 14)
            atr_multiplier: Multiplier for stop-loss (default: 2.0)
            min_consolidation_bars: Minimum bars for consolidation (default: 10)
            min_price_move: Minimum price move for breakout (default: 0.01)
            risk_per_trade: Risk per trade as % of capital (default: 1.0)
            max_trades_per_day: Maximum trades per day (default: 5)
        """
        default_config = {
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_consolidation_bars': 10,
            'min_price_move': 0.01,
            'risk_per_trade': 1.0,
            'max_trades_per_day': 5,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_ma_period': 20
        }
        
        config = {**default_config, **(config or {})}
        super().__init__(config)
        
        # Initialize state
        self.consolidation_range = None
        self.breakout_direction = None
        self.today_trades = 0
        self.last_trade_day = None
        self.logger = logging.getLogger(f'trading.strategy.{self.__class__.__name__}')
        self.logger.info(f"Initialized with config: {config}")
    
    def _reset_daily_metrics(self, current_time: datetime) -> None:
        """Reset daily trading metrics."""
        if self.last_trade_day != current_time.date():
            self.today_trades = 0
            self.last_trade_day = current_time.date()
    
    def _identify_consolidation(self, data: Dict[str, Any]) -> Tuple[float, float, bool]:
        """
        Identify consolidation pattern in price action.
        
        Returns:
            Tuple: (support_level, resistance_level, is_consolidating)
        """
        try:
            closes = np.array([float(x) for x in data['close'][-self.config['min_consolidation_bars']-1:-1]])
            highs = np.array([float(x) for x in data['high'][-self.config['min_consolidation_bars']-1:-1]])
            lows = np.array([float(x) for x in data['low'][-self.config['min_consolidation_bars']-1:-1]])
            
            if len(closes) < self.config['min_consolidation_bars']:
                return 0, 0, False
                
            # Calculate recent price range
            resistance = np.max(highs)
            support = np.min(lows)
            range_size = (resistance - support) / support
            
            # Check if price is in a tight range
            atr = talib.ATR(
                np.array([float(x) for x in data['high'][-self.config['atr_period']-1:-1]]),
                np.array([float(x) for x in data['low'][-self.config['atr_period']-1:-1]]),
                np.array([float(x) for x in data['close'][-self.config['atr_period']-1:-1]]),
                timeperiod=self.config['atr_period']
            )[-1]
            
            # Check if the range is small compared to ATR
            is_tight_range = range_size < (atr * 0.5)
            
            return support, resistance, is_tight_range
            
        except Exception as e:
            self.logger.error(f"Error identifying consolidation: {str(e)}")
            return 0, 0, False
    
    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate technical indicators.
        """
        try:
            closes = np.array([float(x) for x in data.get('close', [])])
            highs = np.array([float(x) for x in data.get('high', [])])
            lows = np.array([float(x) for x in data.get('low', [])])
            volumes = np.array([float(x) for x in data.get('volume', [])])
            
            # Calculate ATR
            atr = talib.ATR(highs, lows, closes, timeperiod=self.config['atr_period'])
            
            # Calculate RSI
            rsi = talib.RSI(closes, timeperiod=self.config['rsi_period'])
            
            # Calculate volume moving average
            volume_ma = talib.SMA(volumes, timeperiod=self.config['volume_ma_period'])
            
            # Update data with indicators
            data.update({
                'atr': atr.tolist(),
                'rsi': rsi.tolist(),
                'volume_ma': volume_ma.tolist()
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            
        return data
    
    def update(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signals based on breakout patterns.
        """
        try:
            # Reset daily metrics if needed
            current_time = datetime.now()
            self._reset_daily_metrics(current_time)
            
            # Check daily trade limit
            if self.today_trades >= self.config['max_trades_per_day']:
                return None
                
            # Calculate indicators
            data = self.calculate_indicators(data)
            
            # Need enough data
            min_bars = max(self.config['min_consolidation_bars'] * 2, 50)
            if len(data.get('close', [])) < min_bars:
                return None
                
            # Get current values
            close = float(data['close'][-1])
            high = float(data['high'][-1])
            low = float(data['low'][-1])
            volume = float(data.get('volume', [0] * len(data['close']))[-1])
            volume_ma = float(data.get('volume_ma', [0] * len(data['close']))[-1])
            rsi = float(data['rsi'][-1])
            atr = float(data['atr'][-1])
            
            # Identify consolidation
            support, resistance, is_consolidating = self._identify_consolidation(data)
            
            # Check for breakout
            if is_consolidating:
                # Check for upside breakout
                if close > resistance and volume > volume_ma * 1.5 and rsi < self.config['rsi_overbought']:
                    stop_loss = support
                    take_profit = close + (2 * (close - stop_loss))  # 2:1 reward:risk
                    
                    self.today_trades += 1
                    return {
                        'symbol': symbol,
                        'direction': 'BUY',
                        'price': close,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'timestamp': current_time.isoformat(),
                        'indicators': {
                            'support': support,
                            'resistance': resistance,
                            'rsi': rsi,
                            'volume': volume,
                            'volume_ma': volume_ma,
                            'atr': atr
                        }
                    }
                    
                # Check for downside breakout
                elif close < support and volume > volume_ma * 1.5 and rsi > self.config['rsi_oversold']:
                    stop_loss = resistance
                    take_profit = close - (2 * (stop_loss - close))  # 2:1 reward:risk
                    
                    self.today_trades += 1
                    return {
                        'symbol': symbol,
                        'direction': 'SELL',
                        'price': close,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'timestamp': current_time.isoformat(),
                        'indicators': {
                            'support': support,
                            'resistance': resistance,
                            'rsi': rsi,
                            'volume': volume,
                            'volume_ma': volume_ma,
                            'atr': atr
                        }
                    }
                    
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            
        return None