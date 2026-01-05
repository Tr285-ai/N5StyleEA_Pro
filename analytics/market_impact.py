import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import linregress
import statsmodels.api as sm

@dataclass
class MarketImpactModel:
    """Models market impact of trades"""
    symbol: str
    window_size: int = 1000  # Number of trades to consider for impact calculation
    decay_factor: float = 0.99  # Exponential decay factor for older trades
    
    def __post_init__(self):
        self.trade_history = []
        self.impact_params = {
            'alpha': 0.1,  # Temporary values, will be calibrated
            'beta': 0.6,
            'gamma': 0.4
        }
    
    def add_trade(self, price: float, quantity: float, timestamp: datetime = None):
        """Record a new trade"""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.trade_history.append({
            'timestamp': timestamp,
            'price': price,
            'quantity': quantity,
            'side': 1 if quantity > 0 else -1,
            'dollar_volume': abs(price * quantity)
        })
        
        # Keep only the most recent trades
        if len(self.trade_history) > self.window_size * 1.5:
            self.trade_history = self.trade_history[-self.window_size:]
    
    def calculate_instantaneous_impact(self, quantity: float, current_price: float) -> float:
        """Calculate immediate price impact of a trade"""
        if not self.trade_history:
            return 0.0
            
        # Calculate volume participation rate
        avg_daily_volume = self._get_average_volume()
        if avg_daily_volume <= 0:
            return 0.0
            
        participation_rate = abs(quantity) / avg_daily_volume
        
        # Calculate impact using power law model: I = α * σ * (Q/V)^β * S^γ
        # Where:
        # I = Impact in price
        # σ = Volatility
        # Q = Order size
        # V = Average daily volume
        # S = Spread
        volatility = self._get_volatility()
        spread = self._get_average_spread()
        
        impact = (self.impact_params['alpha'] * 
                 (volatility * 100) *  # Convert to basis points
                 (participation_rate ** self.impact_params['beta']) *
                 ((spread / current_price) ** self.impact_params['gamma']))
        
        return impact * (1 if quantity > 0 else -1) * current_price / 10000  # Convert bps to price
    
    def estimate_slippage(self, order_quantity: float, current_price: float, 
                         time_horizon: float = 300.0) -> Tuple[float, float]:
        """Estimate expected slippage for an order
        
        Args:
            order_quantity: Size of the order (positive for buy, negative for sell)
            current_price: Current market price
            time_horizon: Time horizon in seconds over which to execute the order
            
        Returns:
            Tuple of (expected_slippage, 95%_confidence_interval)
        """
        if not self.trade_history:
            return 0.0, 0.0
            
        # Calculate participation rate
        avg_volume = self._get_average_volume(time_window=time_horizon)
        if avg_volume <= 0:
            return 0.0, 0.0
            
        participation = abs(order_quantity) / avg_volume
        
        # Get market conditions
        volatility = self._get_volatility()
        spread = self._get_average_spread()
        
        # Simplified model: slippage = a * participation^b * volatility * spread
        # Coefficients would be calibrated from historical data
        a, b = 0.5, 0.7  # Placeholder values
        expected_slippage = a * (participation ** b) * volatility * spread
        
        # Add some noise for confidence interval
        confidence_interval = expected_slippage * 0.3  # 30% of expected slippage
        
        return expected_slippage, confidence_interval
    
    def analyze_liquidity(self, price_levels: List[Tuple[float, float]]) -> Dict:
        """Analyze liquidity at different price levels
        
        Args:
            price_levels: List of (price, quantity) tuples representing the order book
            
        Returns:
            Dictionary with liquidity metrics
        """
        if not price_levels:
            return {}
            
        prices = np.array([p[0] for p in price_levels])
        quantities = np.array([p[1] for p in price_levels])
        mid_price = (prices[0] + prices[-1]) / 2
        
        # Calculate cumulative volume
        cum_volume = np.cumsum(quantities)
        
        # Calculate value at each level
        value = prices * quantities
        cum_value = np.cumsum(value)
        
        # Calculate price impact
        price_impact = (prices - mid_price) / mid_price
        
        # Calculate liquidity metrics
        result = {
            'total_volume': float(cum_volume[-1]) if len(cum_volume) > 0 else 0.0,
            'total_value': float(cum_value[-1]) if len(cum_value) > 0 else 0.0,
            'vwap': float(np.sum(value) / np.sum(quantities)) if np.sum(quantities) > 0 else mid_price,
            'liquidity_imbalance': self._calculate_imbalance(price_levels),
            'order_book_depth': self._calculate_depth(price_levels, mid_price),
            'price_impact_curve': [
                {'price': float(p), 'cum_volume': float(v), 'price_impact': float(pi)}
                for p, v, pi in zip(prices, cum_volume, price_impact)
            ]
        }
        
        return result
    
    def _get_average_volume(self, time_window: float = 86400.0) -> float:
        """Calculate average volume over a time window"""
        if not self.trade_history:
            return 0.0
            
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=time_window)
        
        # Filter trades within the time window
        recent_trades = [t for t in self.trade_history 
                        if t['timestamp'] >= window_start]
        
        if not recent_trades:
            return 0.0
            
        # Calculate total volume (in base currency)
        total_volume = sum(abs(t['dollar_volume']) for t in recent_trades)
        
        # Annualize if needed
        if time_window < 86400:  # Less than a day
            total_volume *= 86400 / time_window
            
        return total_volume
    
    def _get_volatility(self, lookback: int = 100) -> float:
        """Calculate historical volatility"""
        if len(self.trade_history) < 2:
            return 0.0
            
        # Get recent price changes
        prices = [t['price'] for t in self.trade_history[-lookback:]]
        returns = np.diff(np.log(prices))
        
        if len(returns) < 2:
            return 0.0
            
        # Annualized volatility (assuming 252 trading days)
        return np.std(returns) * np.sqrt(252)
    
    def _get_average_spread(self, lookback: int = 100) -> float:
        """Calculate average bid-ask spread"""
        if len(self.trade_history) < 2:
            return 0.0
            
        # In a real implementation, we'd have order book data
        # For now, estimate from trade prices
        prices = [t['price'] for t in self.trade_history[-lookback:]]
        if len(prices) < 2:
            return 0.0
            
        # Simple spread estimation
        price_changes = np.abs(np.diff(prices))
        return np.median(price_changes)
    
    def _calculate_imbalance(self, price_levels: List[Tuple[float, float]]) -> float:
        """Calculate order book imbalance"""
        if not price_levels:
            return 0.0
            
        # Split into bids and asks
        mid_price = (price_levels[0][0] + price_levels[-1][0]) / 2
        bids = [(p, q) for p, q in price_levels if p < mid_price]
        asks = [(p, q) for p, q in price_levels if p > mid_price]
        
        if not bids or not asks:
            return 0.0
            
        # Calculate total volume on each side
        bid_volume = sum(q for _, q in bids)
        ask_volume = sum(q for _, q in asks)
        
        # Calculate imbalance (-1 to 1)
        if bid_volume + ask_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    def _calculate_depth(self, price_levels: List[Tuple[float, float]], 
                        mid_price: float, depth_levels: List[float] = None) -> Dict:
        """Calculate order book depth at different percentage levels"""
        if depth_levels is None:
            depth_levels = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1%
            
        result = {}
        
        for level in depth_levels:
            price_band = mid_price * level
            lower_bound = mid_price - price_band
            upper_bound = mid_price + price_band
            
            # Calculate volume within price band
            volume = sum(q for p, q in price_levels 
                        if lower_bound <= p <= upper_bound)
            
            result[f'depth_{int(level*100)}pct'] = volume
            
        return result
    
    def calibrate_impact_model(self, historical_trades: List[dict]):
        """Calibrate impact model parameters from historical data
        
        Args:
            historical_trades: List of trade dictionaries with 'price', 'quantity', 'timestamp'
        """
        if not historical_trades or len(historical_trades) < 100:
            return  # Not enough data
            
        # Prepare data
        df = pd.DataFrame(historical_trades)
        df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
        df['signed_volume'] = df['quantity'] / df['quantity'].abs().rolling(window=100).mean()
        df = df.dropna()
        
        if len(df) < 50:
            return  # Still not enough data
            
        try:
            # Simple OLS regression for illustration
            # In practice, use more sophisticated methods
            X = sm.add_constant(np.column_stack([
                df['signed_volume'].values,
                df['signed_volume'].abs().values ** 0.5
            ]))
            
            y = df['log_ret'].values
            
            model = sm.OLS(y, X).fit()
            
            # Update model parameters
            if len(model.params) >= 3:  # Make sure we got all coefficients
                self.impact_params = {
                    'alpha': abs(model.params[1]),
                    'beta': 0.5,  # Would come from a separate estimation
                    'gamma': 0.5  # Would come from a separate estimation
                }
                
        except Exception as e:
            print(f"Error calibrating impact model: {e}")

# Example usage
if __name__ == "__main__":
    # Create and test the market impact model
    impact_model = MarketImpactModel("AAPL")
    
    # Simulate some trades
    np.random.seed(42)
    base_price = 150.0
    for i in range(1000):
        price = base_price + np.random.normal(0, 0.1)
        quantity = np.random.lognormal(mean=0, sigma=0.5) * 100
        impact_model.add_trade(price, quantity)
    
    # Test impact calculation
    impact = impact_model.calculate_instantaneous_impact(1000, base_price)
    print(f"Estimated impact for 1000 shares: ${impact:.4f} per share")
    
    # Test slippage estimation
    slippage, confidence = impact_model.estimate_slippage(5000, base_price)
    print(f"Estimated slippage for 5000 shares: ${slippage:.4f} ± {confidence:.4f}")
    
    # Test liquidity analysis
    order_book = [
        (base_price * (1 - 0.001 * i), 1000 * (1 - 0.1 * i)) for i in range(10, 0, -1)
    ] + [
        (base_price, 0)  # Mid price
    ] + [
        (base_price * (1 + 0.001 * i), 1000 * (1 - 0.1 * i)) for i in range(1, 11)
    ]
    
    liquidity = impact_model.analyze_liquidity(order_book)
    print(f"\nLiquidity Analysis:")
    print(f"Total Volume: {liquidity['total_volume']:.2f}")
    print(f"VWAP: ${liquidity['vwap']:.2f}")
    print(f"Order Book Imbalance: {liquidity['liquidity_imbalance']:.4f}")
