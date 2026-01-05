""
Trading System Integration Module

This module integrates the advanced order types, market impact modeling, and risk controls
into the main trading system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json

# Import the new modules
from execution.advanced_orders import AdvancedOrderManager, IcebergOrder, TWAPOrder
from analytics.market_impact import MarketImpactModel
from risk.advanced_controls import (
    Portfolio, Position, PositionLimit, LossLimit, ConcentrationLimit,
    RealTimeRiskMonitor, CrossProductRiskEngine, RiskCheckResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trading_integration')

class TradingSystemIntegration:
    """Main integration class for the trading system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the trading system with configuration"""
        self.config = config or {}
        self.initialized = False
        
        # Core components
        self.order_manager = None
        self.risk_monitor = None
        self.market_impact_model = None
        self.portfolio = None
        self.cross_product_engine = None
        
        # State
        self.symbols = set()
        self.market_data_subscriptions = set()
        
    async def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return
            
        logger.info("Initializing trading system integration...")
        
        # Initialize portfolio
        self.portfolio = Portfolio()
        self.portfolio.cash = self.config.get('initial_capital', 1000000.0)
        
        # Initialize order manager
        dark_pools = self.config.get('dark_pools', [
            {'name': 'Liquidnet', 'fee': 0.0005, 'liquidity': 1000000},
            {'name': 'ITG POSIT', 'fee': 0.0003, 'liquidity': 750000},
            {'name': 'UBS ATS', 'fee': 0.0004, 'liquidity': 500000}
        ])
        self.order_manager = AdvancedOrderManager(dark_pools)
        
        # Initialize market impact model
        self.market_impact_model = {}
        for symbol in self.symbols:
            self.market_impact_model[symbol] = MarketImpactModel(symbol)
        
        # Set up risk limits
        self._setup_risk_limits()
        
        # Initialize cross-product risk engine
        self.cross_product_engine = CrossProductRiskEngine(
            correlation_threshold=self.config.get('correlation_threshold', 0.7)
        )
        
        # Initialize real-time risk monitoring
        self.risk_monitor = RealTimeRiskMonitor(self.portfolio)
        self.risk_monitor.add_alert_handler(self._handle_risk_alert)
        
        # Start components
        self.order_manager.start()
        self.risk_monitor.start()
        
        self.initialized = True
        logger.info("Trading system integration initialized successfully")
    
    def _setup_risk_limits(self):
        """Configure risk limits from config"""
        # Position limits
        position_limits = PositionLimit(
            max_position_size=self.config.get('max_position_size', {}),
            max_notional_value=self.config.get('max_notional_value', {})
        )
        self.portfolio.add_risk_limit(position_limits)
        
        # Loss limits
        loss_limits = LossLimit(
            max_daily_loss_pct=self.config.get('max_daily_loss_pct', 0.05),
            max_daily_loss_abs=self.config.get('max_daily_loss_abs', 100000.0)
        )
        self.portfolio.add_risk_limit(loss_limits)
        
        # Concentration limits
        concentration_limits = ConcentrationLimit(
            max_single_position_pct=self.config.get('max_single_position_pct', 0.3),
            max_sector_exposure=self.config.get('sector_limits', {})
        )
        self.portfolio.add_risk_limit(concentration_limits)
    
    def _handle_risk_alert(self, alert: Dict):
        """Handle risk alerts"""
        logger.warning(f"RISK ALERT: {alert['message']}")
        
        # In a real system, you might want to:
        # 1. Send notifications (email, SMS, etc.)
        # 2. Trigger risk mitigation strategies
        # 3. Log the alert for compliance
        
        # Example: Auto-liquidate positions if daily loss limit is breached
        if alert.get('metadata', {}).get('limit_type') == 'daily_loss_pct':
            current_loss = alert['metadata'].get('current', 0)
            max_loss = alert['metadata'].get('limit', 0.05)
            
            if current_loss > max_loss * 1.5:  # If loss is 50% beyond limit
                logger.critical("Significant loss detected! Triggering emergency liquidation!")
                # self._emergency_liquidate()
    
    async def submit_order(self, order: Dict) -> Dict:
        """Submit a new order with risk checks"""
        if not self.initialized:
            await self.initialize()
        
        # 1. Pre-trade risk checks
        risk_passed, risk_results = self.portfolio.check_risk(order)
        if not risk_passed:
            return {
                'status': 'rejected',
                'reason': 'risk_check_failed',
                'details': [str(r) for r in risk_results if not r.passed]
            }
        
        # 2. Estimate market impact and slippage
        symbol = order.get('symbol')
        if symbol in self.market_impact_model:
            impact = self.market_impact_model[symbol].calculate_instantaneous_impact(
                order.get('quantity', 0),
                order.get('price', 0)
            )
            
            # Adjust order if impact is too high
            if abs(impact) > self.config.get('max_impact_bps', 10) / 10000:  # 10 bps default
                logger.warning(f"High market impact detected: {impact*10000:.1f} bps")
                # Could implement order slicing here
        
        # 3. Route order (to dark pool or regular exchange)
        if order.get('use_dark_pool', False):
            result = self.order_manager.route_to_dark_pool(order)
            if result['success']:
                logger.info(f"Order routed to dark pool: {result['pool']}")
                order.update({
                    'status': 'submitted',
                    'execution_venue': result['pool'],
                    'order_id': result['order_id']
                })
            else:
                logger.warning("Dark pool routing failed, falling back to regular exchange")
                order['execution_venue'] = 'exchange'
        else:
            order['execution_venue'] = 'exchange'
        
        # 4. Submit order through order manager
        try:
            if order.get('order_type') == 'ICEBERG':
                result = self.order_manager.create_iceberg_order(
                    order_id=order.get('order_id', f"iceberg_{int(datetime.utcnow().timestamp())}"),
                    symbol=order['symbol'],
                    side=order['side'],
                    total_quantity=order['quantity'],
                    display_quantity=order.get('display_quantity', order['quantity'] * 0.1),  # 10% display size
                    price=order.get('price'),
                    order_type=order.get('order_type', 'LIMIT')
                )
            elif order.get('order_type') == 'TWAP':
                result = self.order_manager.create_twap_order(
                    order_id=order.get('order_id', f"twap_{int(datetime.utcnow().timestamp())}"),
                    symbol=order['symbol'],
                    side=order['side'],
                    total_quantity=order['quantity'],
                    duration_seconds=order.get('duration_seconds', 3600),  # 1 hour default
                    price_limit=order.get('price')
                )
            else:
                # Regular order
                result = {'success': True, 'order_id': order.get('order_id', f"order_{int(datetime.utcnow().timestamp())}")}
                
            if result['success']:
                order.update({
                    'status': 'submitted',
                    'order_id': result.get('order_id'),
                    'timestamp': datetime.utcnow().isoformat()
                })
                logger.info(f"Order submitted: {order['order_id']}")
                return order
            else:
                return {
                    'status': 'rejected',
                    'reason': 'order_submission_failed',
                    'details': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'reason': 'system_error',
                'details': str(e)
            }
    
    async def update_market_data(self, market_data: Dict[str, Dict]):
        """Update market data for all components"""
        if not self.initialized:
            await self.initialize()
            
        # Update portfolio positions
        self.portfolio.update_market_data({
            symbol: data['price'] 
            for symbol, data in market_data.items() 
            if 'price' in data
        })
        
        # Update market impact models
        for symbol, data in market_data.items():
            if symbol not in self.market_impact_model:
                self.market_impact_model[symbol] = MarketImpactModel(symbol)
                
            if 'trades' in data:
                for trade in data['trades']:
                    self.market_impact_model[symbol].add_trade(
                        price=trade['price'],
                        quantity=trade['quantity'],
                        timestamp=datetime.fromisoformat(trade['timestamp'])
                    )
        
        # Update cross-product risk engine
        returns = {}
        for symbol, data in market_data.items():
            if 'price' in data and 'previous_close' in data and data['previous_close'] > 0:
                returns[symbol] = [data['price'] / data['previous_close'] - 1.0]
        
        if returns:
            self.cross_product_engine.update_returns(returns)
    
    async def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        if not self.initialized:
            await self.initialize()
            
        # Get portfolio metrics
        portfolio_value = self.portfolio.get_portfolio_value()
        daily_pnl = self.portfolio.get_daily_pnl()
        
        # Get position metrics
        positions = {}
        for symbol, position in self.portfolio.positions.items():
            positions[symbol] = {
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': (position.current_price / position.avg_price - 1) * 100 
                                    if position.avg_price > 0 else 0,
                'value': position.quantity * position.current_price
            }
        
        # Get concentration risk
        concentration_risk = self.cross_product_engine.check_concentration_risk(self.portfolio) \
            if self.cross_product_engine else []
        
        return {
            'portfolio': {
                'value': portfolio_value,
                'cash': self.portfolio.cash,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': (daily_pnl / (portfolio_value - daily_pnl)) * 100 
                                if (portfolio_value - daily_pnl) > 0 else 0.0
            },
            'positions': positions,
            'concentration_risk': concentration_risk,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Gracefully shut down all components"""
        logger.info("Shutting down trading system...")
        
        if hasattr(self.order_manager, 'stop'):
            self.order_manager.stop()
            
        if hasattr(self.risk_monitor, 'stop'):
            self.risk_monitor.stop()
        
        self.initialized = False
        logger.info("Trading system shutdown complete")

# Example usage
async def main():
    # Configuration
    config = {
        'initial_capital': 1000000.0,
        'max_daily_loss_pct': 0.05,  # 5%
        'max_daily_loss_abs': 50000.0,  # $50k
        'max_position_size': {
            'AAPL': 1000,  # Max 1000 shares
            'MSFT': 500    # Max 500 shares
        },
        'max_notional_value': {
            'AAPL': 200000,  # $200k max notional
            'MSFT': 150000   # $150k max notional
        },
        'max_single_position_pct': 0.3,  # 30%
        'sector_limits': {
            'TECH': 0.4,      # Max 40% in tech
            'FINANCIALS': 0.3  # Max 30% in financials
        },
        'correlation_threshold': 0.7,
        'dark_pools': [
            {'name': 'Liquidnet', 'fee': 0.0005, 'liquidity': 1000000},
            {'name': 'ITG POSIT', 'fee': 0.0003, 'liquidity': 750000}
        ]
    }
    
    # Initialize trading system
    trading_system = TradingSystemIntegration(config)
    
    try:
        # Start with some market data
        await trading_system.update_market_data({
            'AAPL': {
                'price': 155.0,
                'previous_close': 150.0,
                'volume': 5000000,
                'trades': [
                    {'price': 154.5, 'quantity': 100, 'timestamp': '2023-01-01T10:00:00'},
                    {'price': 155.0, 'quantity': 200, 'timestamp': '2023-01-01T10:01:00'}
                ]
            },
            'MSFT': {
                'price': 300.0,
                'previous_close': 305.0,
                'volume': 3000000,
                'trades': [
                    {'price': 299.5, 'quantity': 150, 'timestamp': '2023-01-01T10:00:30'},
                    {'price': 300.0, 'quantity': 100, 'timestamp': '2023-01-01T10:01:30'}
                ]
            }
        })
        
        # Submit some orders
        orders = [
            # Regular limit order
            {
                'order_id': 'ORD_001',
                'symbol': 'AAPL',
                'side': 'BUY',
                'quantity': 100,
                'price': 154.9,
                'order_type': 'LIMIT',
                'time_in_force': 'GTC'
            },
            # Iceberg order
            {
                'order_id': 'ICEBERG_001',
                'symbol': 'MSFT',
                'side': 'BUY',
                'quantity': 1000,
                'display_quantity': 100,
                'price': 299.5,
                'order_type': 'ICEBERG',
                'time_in_force': 'GTC'
            },
            # TWAP order
            {
                'order_id': 'TWAP_001',
                'symbol': 'AAPL',
                'side': 'SELL',
                'quantity': 500,
                'duration_seconds': 3600,  # 1 hour
                'price': 156.0,
                'order_type': 'TWAP'
            }
        ]
        
        # Process orders
        for order in orders:
            result = await trading_system.submit_order(order)
            print(f"Order result: {json.dumps(result, indent=2)}")
        
        # Get risk metrics
        metrics = await trading_system.get_risk_metrics()
        print("\nRisk Metrics:")
        print(json.dumps(metrics, indent=2))
        
    finally:
        # Clean up
        await trading_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
