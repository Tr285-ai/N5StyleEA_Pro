# risk_manager.py
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Import the existing regime detector
from regime_detector import (
    MarketRegime,
    RegimePrediction,
    RegimeDetector,
    LiquidityLevel
)

class RiskLevel(Enum):
    """Risk levels for position sizing"""
    VERY_LOW = 0.25
    LOW = 0.5
    NORMAL = 1.0
    HIGH = 1.5
    VERY_HIGH = 2.0

@dataclass
class TradeRiskProfile:
    """Container for trade risk assessment"""
    position_size: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    leverage: float
    max_drawdown_pct: float
    risk_level: RiskLevel
    regime: MarketRegime
    confidence: float
    liquidity: LiquidityLevel

class EnhancedRiskManager:
    """
    Enhanced risk management system with regime-aware position sizing
    and trade validation.
    """
    
    def __init__(
        self,
        account_balance: float,
        risk_per_trade: float = 0.01,
        max_drawdown_pct: float = 5.0,
        regime_detector: Optional[RegimeDetector] = None
    ):
        """
        Initialize the risk manager.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Percentage of account to risk per trade (default: 1%)
            max_drawdown_pct: Maximum allowed daily drawdown (default: 5%)
            regime_detector: Optional pre-initialized regime detector
        """
        self.logger = logging.getLogger('risk.manager')
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct
        self.regime_detector = regime_detector or RegimeDetector()
        self.today_pnl = 0.0
        self.max_daily_loss = account_balance * (max_drawdown_pct / 100)
        self.trades_today = 0
        self.max_trades_per_day = 20
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    async def assess_risk(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Any]
    ) -> TradeRiskProfile:
        """
        Assess risk for a potential trade.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss
            take_profit: Proposed take profit
            market_data: Historical market data
            current_positions: Current open positions
            
        Returns:
            TradeRiskProfile with risk assessment
        """
        # Calculate basic risk metrics
        risk_amount = self.account_balance * self.risk_per_trade
        risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        
        # Get market regime
        regime_pred = await self.regime_detector.detect_regime(market_data)
        
        # Adjust position size based on market regime
        position_size = self._calculate_position_size(
            entry_price, stop_loss, risk_amount, regime_pred
        )
        
        # Adjust leverage based on volatility
        leverage = self._calculate_leverage(regime_pred, market_data)
        
        # Check daily limits
        if not self._check_daily_limits():
            position_size = 0.0
            
        # Create risk profile
        return TradeRiskProfile(
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            leverage=leverage,
            max_drawdown_pct=self.max_drawdown_pct,
            risk_level=self._get_risk_level(regime_pred),
            regime=regime_pred.regime,
            confidence=regime_pred.confidence,
            liquidity=self._assess_liquidity(market_data)
        )
        
    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_amount: float,
        regime_pred: RegimePrediction
    ) -> float:
        """Calculate position size based on risk parameters and market regime"""
        # Base position size
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0
            
        position_size = risk_amount / risk_per_share
        
        # Adjust based on market regime
        regime_factor = self._get_regime_factor(regime_pred.regime)
        position_size *= regime_factor
        
        return min(position_size, self._get_max_position_size())
        
    def _calculate_leverage(
        self,
        regime_pred: RegimePrediction,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate appropriate leverage based on market conditions"""
        # Base leverage
        volatility = market_data['close'].pct_change().std() * np.sqrt(252)  # Annualized
        base_leverage = min(5.0, 0.1 / (volatility + 0.01))  # Cap at 5x
        
        # Adjust based on regime
        if regime_pred.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return min(5.0, base_leverage * 1.5)
        elif regime_pred.regime == MarketRegime.VOLATILE:
            return max(1.0, base_leverage * 0.5)
        return base_leverage
        
    def _get_regime_factor(self, regime: MarketRegime) -> float:
        """Get position size multiplier based on market regime"""
        factors = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.1,
            MarketRegime.RANGING: 0.8,
            MarketRegime.VOLATILE: 0.5,
            MarketRegime.UNKNOWN: 0.7
        }
        return factors.get(regime, 1.0)
        
    def _assess_liquidity(self, market_data: pd.DataFrame) -> LiquidityLevel:
        """Assess market liquidity based on volume and spread"""
        avg_volume = market_data['volume'].mean()
        avg_spread = (market_data['high'] - market_data['low']).mean()
        
        if avg_volume > 1000 and avg_spread < 0.0005:
            return LiquidityLevel.HIGH
        elif avg_volume > 500 and avg_spread < 0.001:
            return LiquidityLevel.MEDIUM
        elif avg_volume > 100 and avg_spread < 0.002:
            return LiquidityLevel.LOW
        return LiquidityLevel.NONE
        
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are not exceeded"""
        if self.today_pnl <= -self.max_daily_loss:
            self.logger.warning("Daily loss limit reached")
            return False
        if self.trades_today >= self.max_trades_per_day:
            self.logger.warning("Maximum daily trades reached")
            return False
        return True
        
    def _get_risk_level(self, regime_pred: RegimePrediction) -> RiskLevel:
        """Get risk level based on market regime and confidence"""
        if regime_pred.confidence < 0.5:
            return RiskLevel.VERY_LOW
            
        if regime_pred.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return RiskLevel.HIGH if regime_pred.confidence > 0.7 else RiskLevel.NORMAL
        elif regime_pred.regime == MarketRegime.VOLATILE:
            return RiskLevel.LOW
        return RiskLevel.VERY_LOW
        
    def update_account_balance(self, new_balance: float):
        """Update account balance after a trade"""
        self.account_balance = new_balance
        
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.today_pnl = 0.0
        self.trades_today = 0