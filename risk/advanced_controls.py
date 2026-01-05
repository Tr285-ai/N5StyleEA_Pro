import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import threading
from collections import defaultdict, deque

class RiskCheckResult:
    """Result of a risk check"""
    def __init__(self, passed: bool, message: str = "", metadata: dict = None):
        self.passed = passed
        self.message = message
        self.metadata = metadata or {}
    
    def __bool__(self):
        return self.passed
    
    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"{status}: {self.message}"

class RiskViolationType(Enum):
    POSITION_LIMIT = auto()
    LOSS_LIMIT = auto()
    CONCENTRATION = auto()
    LIQUIDITY = auto()
    VOLATILITY = auto()
    CREDIT = auto()
    COMPLIANCE = auto()

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    timestamp: datetime
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.unrealized_pnl = (new_price - self.avg_price) * self.quantity
        self.current_price = new_price

class RiskLimit:
    """Base class for risk limits"""
    def check(self, portfolio: 'Portfolio', order: dict = None) -> RiskCheckResult:
        raise NotImplementedError
    
    def update(self, market_data: dict):
        """Update risk model with latest market data"""
        pass

@dataclass
class PositionLimit(RiskLimit):
    """Position size limits by symbol"""
    max_position_size: Dict[str, float]  # Symbol -> max position size
    max_notional_value: Dict[str, float]  # Symbol -> max notional value
    
    def check(self, portfolio: 'Portfolio', order: dict = None) -> RiskCheckResult:
        if not order:
            return RiskCheckResult(True)
            
        symbol = order.get('symbol')
        if not symbol:
            return RiskCheckResult(False, "No symbol specified in order")
            
        # Check position size limit
        position = portfolio.positions.get(symbol)
        new_size = (position.quantity if position else 0) + order.get('quantity', 0)
        
        if symbol in self.max_position_size and abs(new_size) > self.max_position_size[symbol]:
            return RiskCheckResult(
                False, 
                f"Position size {new_size} exceeds limit of {self.max_position_size[symbol]} for {symbol}",
                {'limit_type': 'position_size', 'limit': self.max_position_size[symbol], 'current': new_size}
            )
        
        # Check notional value limit
        if symbol in self.max_notional_value:
            notional = abs(new_size * order.get('price', 0))
            if notional > self.max_notional_value[symbol]:
                return RiskCheckResult(
                    False,
                    f"Notional value {notional:,.2f} exceeds limit of {self.max_notional_value[symbol]:,.2f} for {symbol}",
                    {'limit_type': 'notional_value', 'limit': self.max_notional_value[symbol], 'current': notional}
                )
        
        return RiskCheckResult(True)

@dataclass
class LossLimit(RiskLimit):
    """Daily loss limits"""
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_daily_loss_abs: float = 100000.0  # $100k max daily loss
    
    def check(self, portfolio: 'Portfolio', order: dict = None) -> RiskCheckResult:
        # Calculate daily P&L
        daily_pnl = portfolio.get_daily_pnl()
        portfolio_value = portfolio.get_portfolio_value()
        
        # Check percentage loss
        if portfolio_value > 0 and (-daily_pnl / portfolio_value) > self.max_daily_loss_pct:
            loss_pct = (-daily_pnl / portfolio_value) * 100
            return RiskCheckResult(
                False,
                f"Daily loss {loss_pct:.2f}% exceeds limit of {self.max_daily_loss_pct*100:.2f}%",
                {'limit_type': 'daily_loss_pct', 'limit': self.max_daily_loss_pct, 'current': loss_pct/100}
            )
        
        # Check absolute loss
        if -daily_pnl > self.max_daily_loss_abs:
            return RiskCheckResult(
                False,
                f"Daily loss ${-daily_pnl:,.2f} exceeds limit of ${self.max_daily_loss_abs:,.2f}",
                {'limit_type': 'daily_loss_abs', 'limit': self.max_daily_loss_abs, 'current': -daily_pnl}
            )
            
        return RiskCheckResult(True)

@dataclass
class ConcentrationLimit(RiskLimit):
    """Portfolio concentration limits"""
    max_single_position_pct: float = 0.25  # Max 25% in a single position
    max_sector_exposure: Dict[str, float] = None  # Sector -> max exposure
    
    def __post_init__(self):
        if self.max_sector_exposure is None:
            self.max_sector_exposure = {
                'TECH': 0.4,
                'FINANCIALS': 0.3,
                'HEALTHCARE': 0.25,
                'DEFAULT': 0.2
            }
    
    def check(self, portfolio: 'Portfolio', order: dict = None) -> RiskCheckResult:
        portfolio_value = portfolio.get_portfolio_value()
        if portfolio_value <= 0:
            return RiskCheckResult(True)
        
        # Check single position concentration
        for symbol, position in portfolio.positions.items():
            position_value = position.quantity * position.current_price
            if position_value / portfolio_value > self.max_single_position_pct:
                return RiskCheckResult(
                    False,
                    f"Position {symbol} is {position_value/portfolio_value*100:.1f}% of portfolio (max {self.max_single_position_pct*100}%)",
                    {'limit_type': 'position_concentration', 'symbol': symbol, 
                     'current_pct': position_value/portfolio_value, 'limit': self.max_single_position_pct}
                )
        
        # Check sector concentration (simplified)
        # In a real implementation, you'd have sector data for each symbol
        
        return RiskCheckResult(True)

class Portfolio:
    """Represents a trading portfolio with positions"""
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.cash: float = 0.0
        self.trade_history = []
        self.risk_limits: List[RiskLimit] = []
        self._lock = threading.RLock()
    
    def add_risk_limit(self, risk_limit: RiskLimit):
        """Add a risk limit to the portfolio"""
        self.risk_limits.append(risk_limit)
    
    def update_market_data(self, market_data: Dict[str, float]):
        """Update position prices with latest market data"""
        with self._lock:
            for symbol, price in market_data.items():
                if symbol in self.positions:
                    self.positions[symbol].update_price(price)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.cash
        for position in self.positions.values():
            total += position.quantity * position.current_price
        return total
    
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        # In a real implementation, this would filter trades by date
        return sum(trade.get('pnl', 0) for trade in self.trade_history 
                  if trade.get('timestamp', datetime.min) >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    
    def check_risk(self, order: dict = None) -> Tuple[bool, List[RiskCheckResult]]:
        """Check all risk limits"""
        results = []
        with self._lock:
            for risk_limit in self.risk_limits:
                result = risk_limit.check(self, order)
                results.append(result)
                if not result:
                    return False, results
        return True, results

class RealTimeRiskMonitor:
    """Real-time risk monitoring system"""
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.alert_handlers = []
        self.risk_metrics = {}
        self._stop_event = threading.Event()
        self._monitor_thread = None
    
    def start(self):
        """Start the risk monitoring thread"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop(self):
        """Stop the risk monitoring thread"""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
    
    def add_alert_handler(self, handler):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Check all risk limits
                passed, results = self.portfolio.check_risk()
                
                # Update risk metrics
                self._update_metrics()
                
                # Process alerts for any failed checks
                for result in results:
                    if not result.passed:
                        self._trigger_alert(result)
                
                # Sleep briefly to avoid excessive CPU usage
                self._stop_event.wait(1.0)  # Check every second
                
            except Exception as e:
                print(f"Error in risk monitor: {e}")
                self._stop_event.wait(5.0)  # Wait longer on error
    
    def _update_metrics(self):
        """Update risk metrics"""
        portfolio_value = self.portfolio.get_portfolio_value()
        self.risk_metrics = {
            'portfolio_value': portfolio_value,
            'daily_pnl': self.portfolio.get_daily_pnl(),
            'positions': {}
        }
        
        # Calculate position-level metrics
        for symbol, position in self.portfolio.positions.items():
            position_value = position.quantity * position.current_price
            self.risk_metrics['positions'][symbol] = {
                'quantity': position.quantity,
                'value': position_value,
                'weight': position_value / portfolio_value if portfolio_value > 0 else 0,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': (position.unrealized_pnl / (position.avg_price * abs(position.quantity))) * 100 
                                    if position.quantity != 0 and position.avg_price != 0 else 0
            }
    
    def _trigger_alert(self, risk_result: RiskCheckResult):
        """Trigger alert handlers"""
        alert = {
            'timestamp': datetime.utcnow(),
            'message': risk_result.message,
            'severity': 'HIGH',
            'metadata': risk_result.metadata
        }
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")

class CrossProductRiskEngine:
    """Cross-product risk analysis"""
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
        self.position_correlations = {}  # (symbol1, symbol2) -> correlation
        self.historical_returns = {}  # symbol -> list of returns
        self.window_size = 100  # Number of periods for correlation calculation
    
    def update_returns(self, symbol: str, returns: List[float]):
        """Update historical returns for a symbol"""
        if symbol not in self.historical_returns:
            self.historical_returns[symbol] = deque(maxlen=self.window_size)
        
        self.historical_returns[symbol].extend(returns)
        self._update_correlations()
    
    def _update_correlations(self):
        """Update correlation matrix"""
        symbols = list(self.historical_returns.keys())
        n = len(symbols)
        
        for i in range(n):
            for j in range(i+1, n):
                sym1, sym2 = symbols[i], symbols[j]
                returns1 = list(self.historical_returns[sym1])
                returns2 = list(self.historical_returns[sym2])
                
                # Ensure we have enough data and matching lengths
                min_len = min(len(returns1), len(returns2))
                if min_len < 2:
                    continue
                    
                returns1 = returns1[-min_len:]
                returns2 = returns2[-min_len:]
                
                # Calculate correlation
                correlation = np.corrcoef(returns1, returns2)[0, 1]
                self.position_correlations[(sym1, sym2)] = correlation
    
    def check_concentration_risk(self, portfolio: Portfolio) -> List[Dict]:
        """Check for concentrated risk across correlated positions"""
        issues = []
        positions = portfolio.positions
        symbols = list(positions.keys())
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                # Get correlation, default to 0 if not enough data
                correlation = self.position_correlations.get(
                    (sym1, sym2),
                    self.position_correlations.get((sym2, sym1), 0)
                )
                
                if abs(correlation) > self.correlation_threshold:
                    pos1 = positions[sym1]
                    pos2 = positions[sym2]
                    
                    # Both positions in the same direction with high correlation
                    if (pos1.quantity * pos2.quantity) > 0:  # Same sign
                        issues.append({
                            'type': 'correlated_positions',
                            'symbols': [sym1, sym2],
                            'correlation': correlation,
                            'position1': pos1.quantity,
                            'position2': pos2.quantity,
                            'suggestion': 'Consider reducing exposure to one of the correlated positions'
                        })
        
        return issues

# Example usage
if __name__ == "__main__":
    # Create a portfolio
    portfolio = Portfolio()
    portfolio.cash = 1000000.0  # $1M starting cash
    
    # Add some positions
    portfolio.positions['AAPL'] = Position(
        symbol='AAPL',
        quantity=100,
        avg_price=150.0,
        current_price=155.0,
        timestamp=datetime.utcnow()
    )
    
    # Set up risk limits
    position_limits = PositionLimit(
        max_position_size={'AAPL': 1000, 'MSFT': 500},
        max_notional_value={'AAPL': 200000, 'MSFT': 150000}
    )
    
    loss_limits = LossLimit(
        max_daily_loss_pct=0.05,  # 5% max daily loss
        max_daily_loss_abs=50000  # $50k max daily loss
    )
    
    concentration_limits = ConcentrationLimit(
        max_single_position_pct=0.3,  # Max 30% in a single position
        max_sector_exposure={
            'TECH': 0.4,
            'FINANCIALS': 0.3,
            'DEFAULT': 0.2
        }
    )
    
    portfolio.add_risk_limit(position_limits)
    portfolio.add_risk_limit(loss_limits)
    portfolio.add_risk_limit(concentration_limits)
    
    # Set up real-time monitoring
    def alert_handler(alert):
        print(f"\n[ALERT] {alert['timestamp']} - {alert['message']}")
        if 'metadata' in alert:
            print(f"Details: {alert['metadata']}")
    
    monitor = RealTimeRiskMonitor(portfolio)
    monitor.add_alert_handler(alert_handler)
    monitor.start()
    
    try:
        # Simulate some market data updates
        for i in range(5):
            # Update prices
            market_data = {
                'AAPL': 155.0 + i * 2.0,  # AAPL price going up
                'MSFT': 300.0 - i * 1.0   # MSFT price going down
            }
            portfolio.update_market_data(market_data)
            
            # Check risk for a potential order
            order = {
                'symbol': 'AAPL',
                'quantity': 200,  # Try to buy 200 shares
                'price': market_data['AAPL']
            }
            
            # Check if the order would violate any risk limits
            passed, results = portfolio.check_risk(order)
            print(f"\nOrder check: {'APPROVED' if passed else 'REJECTED'}")
            for result in results:
                print(f"- {result}")
            
            # Print portfolio metrics
            print(f"\nPortfolio Value: ${portfolio.get_portfolio_value():,.2f}")
            print(f"Daily P&L: ${portfolio.get_daily_pnl():,.2f}")
            
            # Wait a bit
            import time
            time.sleep(2)
            
    finally:
        monitor.stop()
