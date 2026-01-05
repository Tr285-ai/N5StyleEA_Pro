# File: monitoring/advanced_monitor.py
import time
import json
from typing import Dict, List, Optional, Callable
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TradingMonitor:
    """Advanced monitoring system for trading operations"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.metrics = {}
        self.alerts = []
        self.subscribers = []
        self.performance_metrics = {
            'trades': [],
            'orders': [],
            'executions': [],
            'latency': [],
            'slippage': []
        }
        
    def record_trade(self, trade: dict):
        """Record a new trade"""
        trade['timestamp'] = trade.get('timestamp', datetime.utcnow().isoformat())
        self.performance_metrics['trades'].append(trade)
        self._check_trade_alerts(trade)
        
    def record_order(self, order: dict):
        """Record a new order"""
        order['timestamp'] = order.get('timestamp', datetime.utcnow().isoformat())
        self.performance_metrics['orders'].append(order)
        
    def record_execution(self, execution: dict):
        """Record an execution"""
        execution['timestamp'] = execution.get('timestamp', datetime.utcnow().isoformat())
        self.performance_metrics['executions'].append(execution)
        
    def record_latency(self, event_type: str, latency_ms: float, metadata: dict = None):
        """Record latency metrics"""
        self.performance_metrics['latency'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'latency_ms': latency_ms,
            'metadata': metadata or {}
        })
        
    def record_slippage(self, symbol: str, expected_price: float, 
                       actual_price: float, quantity: float):
        """Record slippage metrics"""
        slippage = (actual_price - expected_price) * quantity
        self.performance_metrics['slippage'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'slippage': slippage,
            'slippage_per_share': slippage / abs(quantity) if quantity else 0,
            'expected_price': expected_price,
            'actual_price': actual_price,
            'quantity': quantity
        })
        
    def subscribe(self, callback: Callable):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback)
        
    def _notify_subscribers(self, event_type: str, data: dict):
        """Notify all subscribers of an event"""
        for subscriber in self.subscribers:
            try:
                subscriber(event_type, data)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
                
    def _check_trade_alerts(self, trade: dict):
        """Check if trade triggers any alerts"""
        # Example: Check for large trades
        if abs(trade.get('quantity', 0) * trade.get('price', 0)) > 100000:  # $100k
            self.trigger_alert(
                'LARGE_TRADE',
                f"Large trade detected: {trade['symbol']} {trade['quantity']} @ {trade['price']}",
                trade
            )
            
    def trigger_alert(self, alert_type: str, message: str, data: dict = None):
        """Trigger a new alert"""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': alert_type,
            'message': message,
            'data': data or {}
        }
        self.alerts.append(alert)
        self._notify_subscribers('ALERT', alert)
        
    def get_performance_summary(self) -> dict:
        """Get summary of performance metrics"""
        trades_df = pd.DataFrame(self.performance_metrics['trades'])
        orders_df = pd.DataFrame(self.performance_metrics['orders'])
        executions_df = pd.DataFrame(self.performance_metrics['executions'])
        latency_df = pd.DataFrame(self.performance_metrics['latency'])
        slippage_df = pd.DataFrame(self.performance_metrics['slippage'])
        
        summary = {
            'total_trades': len(trades_df),
            'total_orders': len(orders_df),
            'total_executions': len(executions_df),
            'avg_latency_ms': latency_df['latency_ms'].mean() if not latency_df.empty else 0,
            'total_slippage': slippage_df['slippage'].sum() if not slippage_df.empty else 0,
            'active_alerts': len([a for a in self.alerts if a.get('resolved') is not True])
        }
        
        return summary
        
    def plot_performance(self) -> go.Figure:
        """Generate performance visualization"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Trades Over Time', 'Execution Latency', 'Slippage'),
            vertical_spacing=0.1
        )
        
        # Trades over time
        if not pd.DataFrame(self.performance_metrics['trades']).empty:
            trades_df = pd.DataFrame(self.performance_metrics['trades'])
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.set_index('timestamp')
            
            fig.add_trace(
                go.Scatter(
                    x=trades_df.index,
                    y=trades_df['price'],
                    mode='markers',
                    name='Trade Price',
                    marker=dict(
                        size=8,
                        color=trades_df['quantity'].apply(lambda x: 'green' if x > 0 else 'red'),
                        opacity=0.7
                    )
                ),
                row=1, col=1
            )
        
        # Execution latency
        if not pd.DataFrame(self.performance_metrics['latency']).empty:
            latency_df = pd.DataFrame(self.performance_metrics['latency'])
            latency_df['timestamp'] = pd.to_datetime(latency_df['timestamp'])
            
            for event_type in latency_df['event_type'].unique():
                df = latency_df[latency_df['event_type'] == event_type]
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['latency_ms'],
                        mode='lines+markers',
                        name=f'Latency - {event_type}'
                    ),
                    row=2, col=1
                )
        
        # Slippage
        if not pd.DataFrame(self.performance_metrics['slippage']).empty:
            slippage_df = pd.DataFrame(self.performance_metrics['slippage'])
            slippage_df['timestamp'] = pd.to_datetime(slippage_df['timestamp'])
            
            fig.add_trace(
                go.Bar(
                    x=slippage_df['timestamp'],
                    y=slippage_df['slippage'],
                    name='Slippage',
                    marker_color='indianred'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Trading Performance Dashboard",
            showlegend=True
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Slippage", row=3, col=1)
        
        return fig