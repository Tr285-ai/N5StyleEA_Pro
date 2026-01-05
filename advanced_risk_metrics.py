import numpy as np
import pandas as pd
from scipy.stats import norm, t, skew, kurtosis
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numba as nb
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class VaRMethod(Enum):
    HISTORICAL = "historical"
    VARIANCE_COVARIANCE = "var_cov"
    MONTE_CARLO = "monte_carlo"
    CONDITIONAL = "conditional"

@dataclass
@dataclass
class RiskReport:
    var: Dict[float, float]  # confidence_level: var_value
    cvar: Dict[float, float]  # confidence_level: cvar_value
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    expected_shortfall: Dict[float, float]
    marginal_var: Optional[Dict[str, float]] = None
    component_var: Optional[Dict[str, float]] = None
    stress_test_results: Optional[Dict[str, float]] = None
    risk_metrics_timestamp: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
class AdvancedRiskMetrics:
    def __init__(self, returns: pd.Series, positions: Optional[Dict[str, float]] = None):
        """
        Initialize AdvancedRiskMetrics with returns and optional positions.
        
        Args:
            returns: Series of returns (daily or intraday)
            positions: Dictionary of positions {asset: value}
        """
        self.returns = returns.dropna()
        self.positions = positions or {}
        self._fit_distribution()
        self.risk_metrics_history = []
        self.last_metrics_update = None
    
    def _fit_distribution(self) -> None:
        """Fit distribution parameters to returns."""
        self.mean = self.returns.mean()
        self.std = self.returns.std()
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns)
        
        # Fit Student's t-distribution
        if len(self.returns) > 1:
            try:
                self.df, self.loc, self.scale = t.fit(self.returns)
            except:
                self.df, self.loc, self.scale = 3.0, self.mean, self.std
        else:
            self.df, self.loc, self.scale = 3.0, self.mean, self.std
    
    def calculate_var(self, confidence_levels: List[float] = None, 
                     method: VaRMethod = VaRMethod.HISTORICAL, 
                     window: int = 252) -> Dict[float, float]:
        """Calculate Value at Risk using specified method."""
        confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        var_results = {}
        
        for cl in confidence_levels:
            if method == VaRMethod.HISTORICAL:
                var_results[cl] = self._var_historical(cl)
            elif method == VaRMethod.VARIANCE_COVARIANCE:
                var_results[cl] = self._var_vc(cl)
            elif method == VaRMethod.MONTE_CARLO:
                var_results[cl] = self._var_monte_carlo(cl)
            elif method == VaRMethod.CONDITIONAL:
                var_results[cl] = self._var_conditional(cl)
                
        return var_results
    
    def _var_historical(self, confidence_level: float) -> float:
        """Calculate historical VaR."""
        return -np.percentile(self.returns, 100 * (1 - confidence_level))
    
    def _var_vc(self, confidence_level: float) -> float:
        """Calculate variance-covariance VaR."""
        return -(self.mean + self.std * norm.ppf(1 - confidence_level))
    
    def _var_conditional(self, confidence_level: float) -> float:
        """Calculate conditional VaR using Cornish-Fisher expansion."""
        z = norm.ppf(1 - confidence_level)
        z_cf = (z + 
               (z**2 - 1) * self.skew / 6 +
               (z**3 - 3*z) * (self.kurt - 3) / 24 -
               (2*z**3 - 5*z) * (self.skew**2) / 36)
        return -(self.mean + self.std * z_cf)
    
    def _var_monte_carlo(self, confidence_level: float, n_simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        sim_returns = np.random.standard_t(
            self.df, 
            size=n_simulations
        ) * self.scale + self.loc
        return -np.percentile(sim_returns, 100 * (1 - confidence_level))
    
    def calculate_expected_shortfall(self, confidence_levels: List[float] = None) -> Dict[float, float]:
        """Calculate Expected Shortfall (CVaR) for given confidence levels."""
        confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        es_results = {}
        
        for cl in confidence_levels:
            var = self._var_historical(cl)
            es = -self.returns[self.returns <= -var].mean()
            es_results[cl] = es if not np.isnan(es) else 0.0
            
        return es_results
    
    def calculate_marginal_var(self, positions: Dict[str, float], 
                              price_returns: Dict[str, float], 
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate Marginal VaR for each position."""
        portfolio_value = sum(positions.values())
        weights = {k: v/portfolio_value for k, v in positions.items()}
        
        # Calculate portfolio return
        port_return = sum(weights[asset] * ret for asset, ret in price_returns.items() 
                         if asset in weights)
        
        # Calculate Marginal VaR
        mvar = {}
        z = norm.ppf(confidence_level)
        
        for asset in positions:
            if asset in price_returns:
                asset_weight = weights[asset]
                cov = np.cov([self.returns, price_returns[asset]], 
                            rowvar=True, ddof=0)[0, 1]
                mvar[asset] = -z * cov / (self.std * np.sqrt(len(self.returns)))
                
        return mvar
    
    def calculate_incremental_var(self, new_position: Dict[str, float], 
                                 existing_positions: Dict[str, float],
                                 price_returns: Dict[str, float],
                                 confidence_level: float = 0.95) -> float:
        """Calculate Incremental VaR for a new position."""
        # Combine positions
        combined_positions = existing_positions.copy()
        for asset, value in new_position.items():
            combined_positions[asset] = combined_positions.get(asset, 0) + value
            
        # Calculate portfolio VaR before and after
        var_before = self.calculate_var([confidence_level])[confidence_level]
        
        # Update positions and recalculate
        self.positions = combined_positions
        self._fit_distribution()
        var_after = self.calculate_var([confidence_level])[confidence_level]
        
        return var_after - var_before
    
    def run_stress_tests(self, scenarios: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """Run stress test scenarios on the portfolio.
        
        Args:
            scenarios: List of scenario definitions. Each scenario should have:
                - name: Scenario name
                - shock: Shock multiplier for returns
                - volatility_shift: Optional volatility shift
                
        Returns:
            Dictionary of {scenario_name: var_value}
        """
        if scenarios is None:
            scenarios = [
                {"name": "Market Crash", "shock": 2.5, "volatility_shift": 3.0},
                {"name": "Flash Crash", "shock": 5.0, "volatility_shift": 5.0},
                {"name": "Volatility Spike", "shock": 1.5, "volatility_shift": 4.0},
                {"name": "Liquidity Crunch", "shock": 2.0, "slippage_multiplier": 2.0}
            ]
            
        results = {}
        for scenario in scenarios:
            shocked_returns = self.returns * scenario.get('shock', 1.0)
            
            # Apply volatility shift if specified
            if 'volatility_shift' in scenario:
                vol_ratio = scenario['volatility_shift'] * (shocked_returns.std() / (self.returns.std() + 1e-9))
                shocked_returns = (shocked_returns - shocked_returns.mean()) * vol_ratio + shocked_returns.mean()
                
            # Calculate VaR for the scenario
            scenario_metrics = AdvancedRiskMetrics(shocked_returns, self.positions)
            var = scenario_metrics.calculate_var([0.95], VaRMethod.HISTORICAL)
            results[scenario['name']] = var[0.95]
            
        return results
        
    def generate_risk_report(self, include_stress_tests: bool = True) -> RiskReport:
        """Generate comprehensive risk report.
        
        Args:
            include_stress_tests: Whether to include stress test results
            
        Returns:
            RiskReport object with all metrics
        """
        confidence_levels = [0.90, 0.95, 0.99]
        
        # Run stress tests if requested
        stress_test_results = None
        if include_stress_tests:
            try:
                stress_test_results = self.run_stress_tests()
            except Exception as e:
                logger.error(f"Error running stress tests: {e}")
                stress_test_results = {"error": str(e)}
        
        # Create and return the report
        report = RiskReport(
            var=self.calculate_var(confidence_levels, VaRMethod.CONDITIONAL),
            cvar=self.calculate_expected_shortfall(confidence_levels),
            max_drawdown=self._calculate_max_drawdown(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            sortino_ratio=self._calculate_sortino_ratio(),
            calmar_ratio=self._calculate_calmar_ratio(),
            expected_shortfall=self.calculate_expected_shortfall(confidence_levels),
            marginal_var=self._calculate_marginal_var() if self.positions else None,
            component_var=self._calculate_component_var() if self.positions else None,
            stress_test_results=stress_test_results,
            risk_metrics_timestamp=datetime.utcnow().isoformat()
        )
        
        # Store in history
        self.risk_metrics_history.append({
            'timestamp': datetime.utcnow(),
            'report': report
        })
        self.last_metrics_update = datetime.utcnow()
        
        # Keep only last 1000 records
        if len(self.risk_metrics_history) > 1000:
            self.risk_metrics_history = self.risk_metrics_history[-1000:]
            
        return report
        
    def monitor_risk_metrics(self, threshold_config: Dict[str, float] = None) -> Dict[str, bool]:
        """Monitor risk metrics against thresholds.
        
        Args:
            threshold_config: Dictionary of metric -> threshold
                Example: {
                    'var_95': 0.05,  # 5% VaR
                    'max_drawdown': 0.1,  # 10% max drawdown
                    'sharpe_ratio': 1.0  # Minimum Sharpe ratio
                }
                
        Returns:
            Dictionary of {metric: is_within_threshold}
        """
        if threshold_config is None:
            threshold_config = {
                'var_95': 0.05,
                'max_drawdown': 0.1,
                'sharpe_ratio': 1.0
            }
            
        report = self.generate_risk_report(include_stress_tests=False)
        results = {}
        
        # Check each threshold
        for metric, threshold in threshold_config.items():
            if metric.startswith('var_'):
                # Handle VaR metrics (e.g., var_95)
                cl = float(metric.split('_')[1]) / 100
                results[metric] = report.var.get(cl, float('inf')) <= threshold
            elif hasattr(report, metric):
                # Handle direct attributes
                value = getattr(report, metric)
                if isinstance(value, (int, float)):
                    results[metric] = value >= threshold if metric == 'sharpe_ratio' else value <= threshold
                    
        return results
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min()
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = self.returns - risk_free_rate/252
        return np.sqrt(252) * (excess_returns.mean() / (excess_returns.std() + 1e-9))
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sortino ratio."""
        excess_returns = self.returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(252) * (excess_returns.mean() / (downside_std + 1e-9))
    
    def _calculate_calmar_ratio(self, years: int = 3) -> float:
        """Calculate Calmar ratio."""
        cagr = self._calculate_cagr(years)
        max_dd = abs(self._calculate_max_drawdown())
        return cagr / (max_dd + 1e-9)
    
    def _calculate_cagr(self, years: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        cum_return = (1 + self.returns).prod() - 1
        return (1 + cum_return) ** (1/years) - 1
    
    def _calculate_marginal_var(self) -> Dict[str, float]:
        """Calculate Marginal VaR for each position.
        
        Returns:
            Dictionary of {asset: marginal_var}
        """
        if not self.positions:
            return {}
            
        # Calculate portfolio VaR
        portfolio_var = self.calculate_var([0.95], VaRMethod.CONDITIONAL)[0.95]
        
        # Calculate component VaR (simplified)
        total_value = sum(abs(v) for v in self.positions.values())
        if total_value == 0:
            return {asset: 0.0 for asset in self.positions}
            
        # Distribute VaR proportionally (simplified)
        return {
            asset: (abs(value) / total_value) * portfolio_var 
            for asset, value in self.positions.items()
        }
    
    def _calculate_component_var(self) -> Dict[str, float]:
        """Calculate Component VaR for each position.
        
        Component VaR shows how much each position contributes to the overall VaR.
        
        Returns:
            Dictionary of {asset: component_var}
        """
        marginal_var = self._calculate_marginal_var()
        if not marginal_var:
            return {}
            
        # Component VaR is marginal VaR * position value
        return {
            asset: marginal_var[asset] * abs(value)
            for asset, value in self.positions.items()
            if asset in marginal_var
        }

# Fast numerical functions with Numba
@nb.njit(fastmath=True, cache=True)
def _calculate_portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

@nb.njit(fastmath=True, cache=True)
def _simulate_var_monte_carlo(returns: np.ndarray, n_simulations: int, 
                            confidence_level: float) -> float:
    """Fast Monte Carlo VaR simulation using Numba."""
    n = len(returns)
    sim_returns = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        idx = np.random.randint(0, n)
        sim_returns[i] = returns[idx]
        
    return -np.percentile(sim_returns, 100 * (1 - confidence_level))
