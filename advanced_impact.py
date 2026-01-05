# File: analytics/advanced_impact.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import statsmodels.api as sm
from scipy.optimize import minimize

@dataclass
class ImpactParameters:
    alpha: float = 0.1
    beta: float = 0.6
    gamma: float = 0.4
    decay_factor: float = 0.99

class AdvancedImpactModel:
    """Advanced market impact modeling with machine learning"""
    
    def __init__(self, symbol: str, params: ImpactParameters = None):
        self.symbol = symbol
        self.params = params or ImpactParameters()
        self.trade_history = []
        self.model = None
        self.features = ['log_volume', 'volatility', 'spread', 'momentum']
        
    def add_trade(self, price: float, quantity: float, timestamp=None, **kwargs):
        """Add a new trade to the impact model"""
        if timestamp is None:
            timestamp = pd.Timestamp.utcnow()
            
        self.trade_history.append({
            'timestamp': timestamp,
            'price': float(price),
            'quantity': float(quantity),
            **kwargs
        })
        
    def calculate_impact(self, quantity: float, current_price: float, 
                        volatility: float = None, spread: float = None) -> float:
        """Calculate expected market impact"""
        if not self.trade_history:
            return 0.0
            
        # Get or calculate market metrics
        volatility = volatility or self._calculate_volatility()
        spread = spread or self._calculate_spread()
        
        # Calculate participation rate
        avg_volume = self._get_average_volume()
        participation = abs(quantity) / avg_volume if avg_volume > 0 else 0
        
        # Calculate impact using power law model
        impact = (self.params.alpha * volatility * 
                 (participation ** self.params.beta) *
                 ((spread / current_price) ** self.params.gamma))
        
        return impact * (1 if quantity > 0 else -1)
    
    def calibrate_model(self, X: pd.DataFrame, y: np.ndarray):
        """Calibrate model using historical data"""
        if len(X) < 10:  # Need sufficient data
            return
            
        try:
            # Add constant for intercept
            X = sm.add_constant(X)
            
            # Fit OLS model
            self.model = sm.OLS(y, X).fit()
            
            # Update parameters
            if len(self.model.params) > 1:
                self.params.alpha = float(abs(self.model.params[0]))
                self.params.beta = float(abs(self.model.params[1]))
                if len(self.model.params) > 2:
                    self.params.gamma = float(abs(self.model.params[2]))
                    
        except Exception as e:
            print(f"Error calibrating model: {e}")
    
    def _calculate_volatility(self, window: int = 20) -> float:
        """Calculate historical volatility"""
        if len(self.trade_history) < 2:
            return 0.0
            
        prices = pd.Series([t['price'] for t in self.trade_history])
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_spread(self, window: int = 10) -> float:
        """Calculate average spread"""
        if not self.trade_history:
            return 0.0
            
        # In a real implementation, use order book data
        # This is a simplified version
        prices = [t['price'] for t in self.trade_history[-window:]]
        if len(prices) < 2:
            return 0.0
        return np.mean(np.abs(np.diff(prices)))
    
    def _get_average_volume(self, window: int = 20) -> float:
        """Calculate average trading volume"""
        if not self.trade_history:
            return 0.0
            
        volumes = [abs(t['quantity']) for t in self.trade_history[-window:]]
        return np.mean(volumes) if volumes else 0.0