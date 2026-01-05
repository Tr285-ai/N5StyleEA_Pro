import asyncio
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .hft_optimizer import HFTOptimizer, HFTOptimizationParams
from .advanced_risk_metrics import AdvancedRiskMetrics, RiskReport, VaRMethod
from .trading_strategy import TradingStrategy

logger = logging.getLogger(__name__)

class HFTRiskIntegrator:
    """
    Integrates HFT optimizations and advanced risk management into the trading system.
    """
    
    def __init__(
        self, 
        trading_strategy: TradingStrategy,
        hft_params: Optional[Dict] = None,
        risk_params: Optional[Dict] = None
    ):
        """
        Initialize the HFT and risk integration layer.
        
        Args:
            trading_strategy: Instance of the main trading strategy
            hft_params: Parameters for HFT optimization
            risk_params: Parameters for risk management
        """
        self.strategy = trading_strategy
        
        # Initialize HFT Optimizer
        hft_params = hft_params or {}
        self.hft_optimizer = HFTOptimizer(
            HFTOptimizationParams(**hft_params)
        )
        
        # Initialize risk metrics
        self.risk_metrics = None
        self.risk_report = None
        self.positions = {}
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
    async def process_market_data(
        self, 
        market_data: pd.DataFrame, 
        order_book: Dict[str, np.ndarray],
        symbol: str
    ) -> Optional[Dict]:
        """
        Process market data with HFT optimizations and risk checks.
        
        Args:
            market_data: OHLCV market data
            order_book: Current order book snapshot
            symbol: Trading pair symbol
            
        Returns:
            Trading signal with HFT optimizations or None if no trade
        """
        # Update order book
        self.hft_optimizer.update_order_book(order_book)
        
        # Generate base signal from strategy
        signal = await self.strategy.generate_signal(market_data, symbol)
        if not signal or signal.get('action') == 'hold':
            return None
            
        # Apply HFT optimizations
        optimized_order = self._optimize_order_execution(
            signal['action'],
            signal.get('quantity', 0),
            market_data
        )
        
        # Apply risk management
        if not self._check_risk_limits(signal, market_data):
            return None
            
        # Update trade history
        self._update_trade_history(signal, market_data)
        
        # Update risk metrics
        self._update_risk_metrics(market_data)
        
        return {
            **signal,
            'optimized_price': optimized_order['price'],
            'optimized_quantity': optimized_order['size'],
            'market_impact': optimized_order['market_impact'],
            'risk_metrics': self.risk_report.dict() if self.risk_report else {}
        }
    
    def _optimize_order_execution(
        self, 
        action: str, 
        quantity: float,
        market_data: pd.DataFrame
    ) -> Dict:
        """Optimize order execution using HFT techniques."""
        start_time = datetime.utcnow()
        
        # Get optimized order parameters
        optimized_order = self.hft_optimizer.optimize_order_execution(
            action,
            quantity
        )
        
        # Measure and log latency
        self.hft_optimizer.measure_latency(start_time.timestamp())
        
        return optimized_order
    
    def _check_risk_limits(
        self, 
        signal: Dict, 
        market_data: pd.DataFrame
    ) -> bool:
        """Check if trade meets risk limits."""
        # Update positions
        self._update_positions(signal)
        
        # Check position limits
        if not self._check_position_limits(signal):
            return False
            
        # Check VaR limits
        if not self._check_var_limits(market_data):
            return False
            
        return True
    
    def _update_positions(self, signal: Dict) -> None:
        """Update internal position tracking."""
        symbol = signal.get('symbol')
        quantity = signal.get('quantity', 0)
        action = signal.get('action')
        
        if action == 'buy':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        elif action == 'sell':
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
    
    def _check_position_limits(self, signal: Dict) -> bool:
        """Check if position is within limits."""
        # Implement position limit checks here
        # For example, max position size, max exposure per symbol, etc.
        return True
    
    def _check_var_limits(self, market_data: pd.DataFrame) -> bool:
        """Check if trade is within VaR limits."""
        if self.risk_report is None:
            return True
            
        # Check against VaR limits
        var_95 = self.risk_report.var.get(0.95, 0)
        max_var = 0.05  # 5% of portfolio value
        
        return var_95 <= max_var
    
    def _update_trade_history(self, signal: Dict, market_data: pd.DataFrame) -> None:
        """Update trade history with new trade."""
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'symbol': signal.get('symbol'),
            'action': signal.get('action'),
            'quantity': signal.get('quantity'),
            'price': market_data['close'].iloc[-1],
            'pnl': 0,  # Will be updated when position is closed
            'status': 'open'
        })
    
    def _update_risk_metrics(self, market_data: pd.DataFrame) -> None:
        """Update risk metrics with latest market data."""
        # Calculate returns
        returns = market_data['close'].pct_change().dropna()
        
        # Update risk metrics
        self.risk_metrics = AdvancedRiskMetrics(
            returns=returns,
            positions=self.positions
        )
        
        # Generate risk report
        self.risk_report = self.risk_metrics.generate_risk_report()
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        if self.risk_report is None:
            return {}
            
        return {
            'sharpe_ratio': self.risk_report.sharpe_ratio,
            'sortino_ratio': self.risk_report.sortino_ratio,
            'max_drawdown': self.risk_report.max_drawdown,
            'var_95': self.risk_report.var.get(0.95, 0),
            'cvar_95': self.risk_report.cvar.get(0.95, 0),
            'open_positions': len([p for p in self.positions.values() if p != 0]),
            'total_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self.trade_history:
            return 0.0
            
        closed_trades = [t for t in self.trade_history if t['status'] == 'closed']
        if not closed_trades:
            return 0.0
            
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(closed_trades)

# Example usage
if __name__ == "__main__":
    # Initialize components
    strategy = TradingStrategy(model_path="models/ensemble_model.h5")
    integrator = HFTRiskIntegrator(strategy)
    
    # Example market data
    market_data = pd.DataFrame({
        'open': [100, 101, 102, 101, 103],
        'high': [101, 102, 103, 103, 104],
        'low': [99, 100, 101, 100, 102],
        'close': [100.5, 101.5, 102.5, 102, 103.5],
        'volume': [1000, 1200, 1500, 1300, 1600]
    })
    
    # Example order book
    order_book = {
        'bids': np.array([
            [100.4, 10],
            [100.3, 15],
            [100.2, 20]
        ]),
        'asks': np.array([
            [100.6, 12],
            [100.7, 18],
            [100.8, 25]
        ])
    }
    
    # Process market data
    async def main():
        signal = await integrator.process_market_data(
            market_data, 
            order_book,
            'BTC/USD'
        )
        print("Generated signal:", signal)
        print("Performance metrics:", integrator.get_performance_metrics())
    
    asyncio.run(main())
