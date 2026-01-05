# File: monitoring/advanced_monitoring.py
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import asyncio
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class AlertLevel:
    INFO = 'info'
    WARNING = 'warning'
    CRITICAL = 'critical'

@dataclass
class Alert:
    id: str
    level: str
    message: str
    timestamp: float
    metadata: dict = field(default_factory=dict)
    acknowledged: bool = False

class AdvancedMonitor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.metrics = {
            'latency': [],
            'slippage': [],
            'fill_rate': [],
            'reject_rate': []
        }
        self.trade_history = []
        self.performance_metrics = {}
        self.alert_handlers = []
        
    def record_trade(self, trade: Dict) -> None:
        """Record a completed trade"""
        trade['timestamp'] = time.time()
        self.trade_history.append(trade)
        self._update_performance_metrics()
        
    def record_latency(self, event_type: str, latency_ms: float) -> None:
        """Record latency metric"""
        self.metrics['latency'].append({
            'timestamp': time.time(),
            'event_type': event_type,
            'latency_ms': latency_ms
        })
        
    def record_slippage(self, symbol: str, expected: float, 
                       actual: float, quantity: float) -> None:
        """Record slippage metric"""
        slippage = (actual - expected) / expected if expected else 0
        self.metrics['slippage'].append({
            'timestamp': time.time(),
            'symbol': symbol,
            'slippage_pct': slippage * 100,
            'quantity': quantity
        })
        
    def trigger_alert(self, level: str, message: str, 
                     metadata: Optional[Dict] = None) -> str:
        """Trigger a new alert"""
        alert = Alert(
            id=f"alert_{int(time.time())}",
            level=level,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.alerts.append(alert)
        self._notify_handlers(alert)
        return alert.id
        
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on trade history"""
        if not self.trade_history:
            return
            
        df = pd.DataFrame(self.trade_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        # Calculate basic metrics
        self.performance_metrics = {
            'total_trades': len(df),
            'win_rate': (df['pnl'] > 0).mean() if 'pnl' in df else 0,
            'avg_trade': df['pnl'].mean() if 'pnl' in df else 0,
            'max_drawdown': self._calculate_max_drawdown(df['pnl'].cumsum() if 'pnl' in df else pd.Series([0])),
            'sharpe_ratio': self._calculate_sharpe_ratio(df['pnl'] if 'pnl' in df else pd.Series([0]))
        }
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series"""
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min() * 100  # Return as percentage
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-9)
        
    def _notify_handlers(self, alert: Alert) -> None:
        """Notify all registered alert handlers"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
    
    def get_performance_dashboard(self) -> go.Figure:
        """Generate performance dashboard"""
        if not self.trade_history:
            return None
            
        df = pd.DataFrame(self.trade_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Equity Curve', 'Daily Returns', 'Trade Distribution'),
            vertical_spacing=0.1
        )
        
        # Equity Curve
        if 'pnl' in df:
            equity = df.set_index('timestamp')['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(x=equity.index, y=equity, name='Equity'),
                row=1, col=1
            )
            
        # Daily Returns
        if 'pnl' in df:
            daily_returns = df.set_index('timestamp')['pnl'].resample('D').sum()
            fig.add_trace(
                go.Bar(x=daily_returns.index, y=daily_returns, name='Daily Returns'),
                row=2, col=1
            )
            
        # Trade Distribution
        if 'pnl' in df:
            fig.add_trace(
                go.Histogram(x=df['pnl'], name='Trade P&L Distribution'),
                row=3, col=1
            )
            
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Trading Performance Dashboard"
        )
        
        return fig