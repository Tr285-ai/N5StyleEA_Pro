import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy
from ..models import Order, OrderSide, OrderType

class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy that identifies trends using RSI and MACD.
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        default_config = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 1.0,
            'min_trend_strength': 25,
            'max_trades_per_day': 5
        }
        super().__init__(config or {})
        self.config = {**default_config, **self.config}
        self.today_trades = 0
        self.last_trade_day = None
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the strategy."""
        self.initialized = True
        
    async def on_market_data(self, data: Dict[str, Any]) -> List[Order]:
        """Process market data and generate trading signals."""
        if not self._validate_market_data(data):
            return []
            
        try:
            indicators = self._calculate_indicators(data)
            return self._generate_signals(data, indicators)
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}", exc_info=True)
            return []
    
    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming market data."""
        if not data or 'close' not in data or not data['close']:
            return False
        return True
    
    def _calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators."""
        closes = pd.Series([float(x) for x in data.get('close', [])])
        highs = pd.Series([float(x) for x in data.get('high', [])])
        lows = pd.Series([float(x) for x in data.get('low', [])])
        
        indicators = {
            'rsi': self._calculate_rsi(closes),
            'macd': self._calculate_macd(closes),
            'atr': self._calculate_atr(highs, lows, closes),
            'adx': self._calculate_adx(highs, lows, closes)
        }
        
        return indicators
    
    def _calculate_rsi(self, closes: pd.Series) -> float:
        """Calculate RSI indicator."""
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    def _calculate_macd(self, closes: pd.Series) -> Dict[str, float]:
        """Calculate MACD indicator."""
        macd, signal, _ = talib.MACD(
            closes,
            fastperiod=self.config['macd_fast'],
            slowperiod=self.config['macd_slow'],
            signalperiod=self.config['macd_signal']
        )
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1]
        }
    
    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
        """Calculate ATR indicator."""
        return talib.ATR(
            highs, 
            lows, 
            closes, 
            timeperiod=self.config['atr_period']
        ).iloc[-1]
    
    def _calculate_adx(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
        """Calculate ADX indicator."""
        return talib.ADX(
            highs,
            lows,
            closes,
            timeperiod=14
        ).iloc[-1]
    
    def _generate_signals(self, data: Dict[str, Any], indicators: Dict[str, Any]) -> List[Order]:
        """Generate trading signals based on indicators."""
        orders = []
        current_rsi = indicators['rsi']
        macd = indicators['macd']
        adx = indicators['adx']
        
        # Check trend strength
        if adx < self.config['min_trend_strength']:
            return orders
            
        # Check RSI conditions
        if current_rsi < self.config['rsi_oversold'] and macd['macd'] > macd['signal']:
            orders.append(Order(
                symbol=data.get('symbol', 'UNKNOWN'),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0
            ))
        elif current_rsi > self.config['rsi_overbought'] and macd['macd'] < macd['signal']:
            orders.append(Order(
                symbol=data.get('symbol', 'UNKNOWN'),
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=1.0
            ))
            
        return orders
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for this strategy."""
        base_schema = super().get_config_schema()
        return {
            **base_schema,
            "rsi_period": {
                "type": "integer",
                "default": 14,
                "min": 2,
                "max": 50,
                "description": "RSI period"
            },
            "rsi_overbought": {
                "type": "float",
                "default": 70.0,
                "min": 50.0,
                "max": 90.0,
                "description": "RSI overbought threshold"
            },
            "rsi_oversold": {
                "type": "float",
                "default": 30.0,
                "min": 10.0,
                "max": 50.0,
                "description": "RSI oversold threshold"
            },
            "macd_fast": {
                "type": "integer",
                "default": 12,
                "min": 1,
                "max": 50,
                "description": "MACD fast period"
            },
            "macd_slow": {
                "type": "integer",
                "default": 26,
                "min": 2,
                "max": 100,
                "description": "MACD slow period"
            },
            "macd_signal": {
                "type": "integer",
                "default": 9,
                "min": 1,
                "max": 50,
                "description": "MACD signal period"
            },
            "min_trend_strength": {
                "type": "float",
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "description": "Minimum ADX value for trend"
            }
        }