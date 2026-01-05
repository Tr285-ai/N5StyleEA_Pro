"""
simulate_backtest_orchestrator.py

A comprehensive backtesting orchestrator for the trading system that integrates with
the existing components including micro-predictor, sentiment analysis, and pattern recognition.
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger('backtest_orchestrator')

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import local modules
try:
    from data_feed_v15_2 import DataFeedV15_2, MarketData
    from micro_predictor_v15_2 import MicroPredictor, PredictionResult
    from sentiment_engine_v15_2 import SentimentEngine, SentimentResult
    from pattern_engine_v15_2 import PatternEngine, PatternResult
    from expiry_advanced_v15_2 import ExpirySelector
    from ml_core_v15_2 import predict_direction, predict_regime
    from trade_api_v15_2 import TradeAPI, OrderType, OrderSide, OrderStatus
    from config import load_config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_open_trades: int = 5
    commission: float = 0.0005  # 0.05% commission per trade
    slippage: float = 0.0001  # 0.01% slippage
    enable_micro_predictor: bool = True
    enable_sentiment: bool = True
    enable_patterns: bool = True
    enable_ml: bool = True
    output_dir: str = "backtest_results"
    random_seed: Optional[int] = 42

@dataclass
class TradeRecord:
    """Record of a single trade."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    position_size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    side: str = "LONG"  # or "SHORT"
    status: str = "OPEN"  # or "CLOSED", "STOPPED", "TAKE_PROFIT"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: float = 2.0
    confidence: float = 0.0
    metadata: dict = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary for serialization."""
        result = asdict(self)
        result['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        result['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return result

class BacktestOrchestrator:
    """Main orchestrator for backtesting trading strategies."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize the backtest orchestrator."""
        self.config = config
        self.data_feed = None
        self.micro_predictor = None
        self.sentiment_engine = None
        self.pattern_engine = None
        self.expiry_selector = None
        self.trade_api = None
        self.trades: List[TradeRecord] = []
        self.equity_curve = []
        self.current_balance = config.initial_balance
        self.open_trades: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []
        self.metrics = {}
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        logger.info("Initializing backtest components...")
        
        # Initialize data feed
        self.data_feed = DataFeedV15_2(
            symbols=[self.config.symbol],
            timeframes=[self.config.timeframe],
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        # Initialize micro predictor if enabled
        if self.config.enable_micro_predictor:
            self.micro_predictor = MicroPredictor()
            logger.info("Micro predictor initialized")
        
        # Initialize sentiment engine if enabled
        if self.config.enable_sentiment:
            self.sentiment_engine = SentimentEngine()
            logger.info("Sentiment engine initialized")
        
        # Initialize pattern engine if enabled
        if self.config.enable_patterns:
            self.pattern_engine = PatternEngine()
            logger.info("Pattern engine initialized")
        
        # Initialize expiry selector
        self.expiry_selector = ExpirySelector()
        
        # Initialize trade API
        self.trade_api = TradeAPI(
            initial_balance=self.config.initial_balance,
            commission=self.config.commission,
            slippage=self.config.slippage
        )
        
        logger.info("All components initialized successfully")
    
    def run(self) -> Dict[str, Any]:
        """Run the backtest."""
        logger.info(f"Starting backtest for {self.config.symbol} from {self.config.start_date} to {self.config.end_date}")
        start_time = time.time()
        
        try:
            # Main backtest loop
            for candle in tqdm(self.data_feed.get_candles(), desc="Processing candles"):
                self._process_candle(candle)
            
            # Close any remaining open trades
            self._close_all_trades()
            
            # Calculate performance metrics
            self._calculate_metrics()
            
            # Generate reports
            self._generate_reports()
            
            logger.info(f"Backtest completed in {time.time() - start_time:.2f} seconds")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error during backtest: {e}", exc_info=True)
            raise
    
    def _process_candle(self, candle: Dict[str, Any]) -> None:
        """Process a single candle."""
        # Update current price
        current_price = candle['close']
        current_time = candle['timestamp']
        
        # Update open trades
        self._update_open_trades(current_price, current_time)
        
        # Check for new trade signals if we have capacity
        if len(self.open_trades) < self.config.max_open_trades:
            signal = self._generate_signal(candle)
            if signal:
                self._execute_trade(signal, current_price, current_time)
        
        # Update equity curve
        self._update_equity_curve(current_price, current_time)
    
    def _generate_signal(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a trading signal based on current market conditions."""
        signal = {
            'symbol': self.config.symbol,
            'timestamp': candle['timestamp'],
            'price': candle['close'],
            'confidence': 0.0,
            'side': None,  # 'BUY' or 'SELL'
            'stop_loss': None,
            'take_profit': None,
            'features': {}
        }
        
        # Get micro predictions if enabled
        if self.config.enable_micro_predictor:
            try:
                prediction = self.micro_predictor.predict(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    current_price=candle['close']
                )
                signal['confidence'] = prediction.confidence
                signal['side'] = 'BUY' if prediction.direction > 0 else 'SELL'
                signal['features']['micro_prediction'] = prediction.to_dict()
            except Exception as e:
                logger.warning(f"Error getting micro prediction: {e}")
        
        # Get sentiment if enabled
        if self.config.enable_sentiment:
            try:
                sentiment = self.sentiment_engine.analyze(symbol=self.config.symbol)
                signal['features']['sentiment'] = sentiment.to_dict()
                # Adjust confidence based on sentiment
                if sentiment.overall_sentiment == 'bullish':
                    signal['confidence'] *= 1.2
                elif sentiment.overall_sentiment == 'bearish':
                    signal['confidence'] *= 0.8
            except Exception as e:
                logger.warning(f"Error getting sentiment: {e}")
        
        # Get patterns if enabled
        if self.config.enable_patterns:
            try:
                patterns = self.pattern_engine.analyze(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe
                )
                signal['features']['patterns'] = [p.to_dict() for p in patterns]
                
                # Adjust signal based on patterns
                for pattern in patterns:
                    if pattern.confidence > 0.7:  # Strong pattern
                        if pattern.pattern_type in ['bullish_engulfing', 'morning_star']:
                            signal['side'] = 'BUY'
                            signal['confidence'] = max(signal['confidence'], 0.7)
                        elif pattern.pattern_type in ['bearish_engulfing', 'evening_star']:
                            signal['side'] = 'SELL'
                            signal['confidence'] = max(signal['confidence'], 0.7)
            except Exception as e:
                logger.warning(f"Error getting patterns: {e}")
        
        # Get ML predictions if enabled
        if self.config.enable_ml:
            try:
                # Prepare features for ML model
                features = {
                    'close': candle['close'],
                    'volume': candle['volume'],
                    'volatility': (candle['high'] - candle['low']) / candle['close'],
                    'rsi': self._calculate_rsi(candle),
                    'macd': self._calculate_macd(candle),
                    # Add more features as needed
                }
                
                # Get direction prediction
                direction_pred = predict_direction(features)
                regime_pred = predict_regime(features)
                
                signal['features']['ml'] = {
                    'direction': direction_pred,
                    'regime': regime_pred
                }
                
                # Adjust signal based on ML predictions
                if direction_pred > 0.6:  # Strong buy signal
                    signal['side'] = 'BUY'
                    signal['confidence'] = max(signal['confidence'], direction_pred)
                elif direction_pred < 0.4:  # Strong sell signal
                    signal['side'] = 'SELL'
                    signal['confidence'] = max(signal['confidence'], 1 - direction_pred)
                
                # Adjust position sizing based on regime
                if regime_pred == 'high_volatility':
                    signal['position_size'] = 0.5  # Reduce position size
                elif regime_pred == 'low_volatility':
                    signal['position_size'] = 1.5  # Increase position size
                
            except Exception as e:
                logger.warning(f"Error getting ML predictions: {e}")
        
        # Only return signals that meet minimum confidence threshold
        if signal['side'] and signal['confidence'] >= 0.6:  # 60% confidence threshold
            # Calculate stop loss and take profit
            atr = self._calculate_atr(candle)
            if signal['side'] == 'BUY':
                signal['stop_loss'] = candle['low'] - atr * 1.5
                signal['take_profit'] = candle['close'] + (atr * 1.5 * 2)  # 1:2 risk-reward
            else:  # SELL
                signal['stop_loss'] = candle['high'] + atr * 1.5
                signal['take_profit'] = candle['close'] - (atr * 1.5 * 2)
            
            return signal
        
        return None
    
    def _execute_trade(self, signal: Dict[str, Any], price: float, timestamp: datetime) -> None:
        """Execute a trade based on the signal."""
        try:
            # Calculate position size based on risk
            if signal['side'] == 'BUY':
                risk_amount = self.current_balance * self.config.risk_per_trade
                stop_distance = price - signal['stop_loss']
                position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            else:  # SELL
                risk_amount = self.current_balance * self.config.risk_per_trade
                stop_distance = signal['stop_loss'] - price
                position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            if position_size <= 0:
                return
            
            # Create trade record
            trade = TradeRecord(
                trade_id=f"TRADE_{len(self.trades) + 1}",
                symbol=signal['symbol'],
                entry_time=timestamp,
                entry_price=price,
                position_size=position_size,
                side=signal['side'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                risk_reward_ratio=2.0,  # Fixed for now, can be dynamic
                confidence=signal['confidence'],
                metadata={
                    'features': signal.get('features', {}),
                    'strategy': 'composite'
                }
            )
            
            # Add to open trades
            self.open_trades[trade.trade_id] = trade
            self.trades.append(trade)
            
            logger.info(f"Opened {trade.side} trade {trade.trade_id} at {price:.5f}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
    
    def _update_open_trades(self, current_price: float, current_time: datetime) -> None:
        """Update open trades and check for exit conditions."""
        to_remove = []
        
        for trade_id, trade in self.open_trades.items():
            if trade.status != 'OPEN':
                continue
                
            # Check stop loss
            if ((trade.side == 'LONG' and current_price <= trade.stop_loss) or
                (trade.side == 'SHORT' and current_price >= trade.stop_loss)):
                self._close_trade(trade, current_price, current_time, 'STOP_LOSS')
                to_remove.append(trade_id)
                continue
                
            # Check take profit
            if ((trade.side == 'LONG' and current_price >= trade.take_profit) or
                (trade.side == 'SHORT' and current_price <= trade.take_profit)):
                self._close_trade(trade, current_price, current_time, 'TAKE_PROFIT')
                to_remove.append(trade_id)
                continue
        
        # Remove closed trades
        for trade_id in to_remove:
            self.open_trades.pop(trade_id, None)
    
    def _close_trade(self, trade: TradeRecord, price: float, timestamp: datetime, reason: str) -> None:
        """Close a trade and calculate P&L."""
        trade.exit_price = price
        trade.exit_time = timestamp
        trade.status = 'CLOSED'
        trade.close_reason = reason
        
        # Calculate P&L
        if trade.side == 'LONG':
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
        else:  # SHORT
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
        
        trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.position_size)) * 100
        
        # Update balance
        self.current_balance += trade.pnl
        
        # Add to closed trades
        self.closed_trades.append(trade)
        
        logger.info(
            f"Closed {trade.side} trade {trade.trade_id} at {price:.5f} "
            f"(P&L: {trade.pnl:.2f} {trade.pnl_pct:.2f}%) - {reason}"
        )
    
    def _close_all_trades(self) -> None:
        """Close all open trades at the current price."""
        if not self.open_trades:
            return
            
        logger.info(f"Closing all {len(self.open_trades)} open trades...")
        current_price = self.data_feed.get_current_price()
        current_time = datetime.utcnow()
        
        for trade in list(self.open_trades.values()):
            self._close_trade(trade, current_price, current_time, 'FORCE_CLOSE')
        
        self.open_trades.clear()
    
    def _update_equity_curve(self, price: float, timestamp: datetime) -> None:
        """Update the equity curve with the current portfolio value."""
        # Calculate open trade P&L
        open_pnl = 0.0
        for trade in self.open_trades.values():
            if trade.side == 'LONG':
                open_pnl += (price - trade.entry_price) * trade.position_size
            else:  # SHORT
                open_pnl += (trade.entry_price - price) * trade.position_size
        
        # Record equity point
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance + open_pnl,
            'open_trades': len(self.open_trades)
        })
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics for the backtest."""
        if not self.closed_trades:
            logger.warning("No closed trades to analyze")
            return
        
        # Basic metrics
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.closed_trades)
        avg_win = (sum(t.pnl for t in self.closed_trades if t.pnl > 0) / 
                  winning_trades) if winning_trades > 0 else 0
        avg_loss = (abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0)) / 
                   losing_trades) if losing_trades > 0 else 0
        profit_factor = (sum(t.pnl for t in self.closed_trades if t.pnl > 0) / 
                        abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0))) if losing_trades > 0 else float('inf')
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600  # in hours
                    for t in self.closed_trades if t.exit_time and t.entry_time]
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
        
        # Store metrics
        self.metrics = {
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'initial_balance': self.config.initial_balance,
            'final_balance': self.current_balance,
            'total_return': ((self.current_balance / self.config.initial_balance) - 1) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration_hours': avg_trade_duration,
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0
            
        peak = self.equity_curve[0]['balance']
        max_dd = 0.0
        
        for point in self.equity_curve:
            if point['balance'] > peak:
                peak = point['balance']
            dd = (peak - point['balance']) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd * 100  # as percentage
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
            
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i]['balance'] / self.equity_curve[i-1]['balance']) - 1
            returns.append(ret)
        
        if not returns:
            return 0.0
            
        excess_returns = [r - risk_free_rate / 252 for r in returns]  # 252 trading days
        avg_excess_return = sum(excess_returns) / len(excess_returns)
        std_dev = (sum((x - avg_excess_return) ** 2 for x in excess_returns) / len(excess_returns)) ** 0.5
        
        return (avg_excess_return / std_dev) * (252 ** 0.5) if std_dev != 0 else 0.0
    
    def _generate_reports(self) -> None:
        """Generate backtest reports and visualizations."""
        if not self.closed_trades:
            logger.warning("No closed trades to generate reports")
            return
        
        # Create reports directory
        report_dir = os.path.join(self.config.output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_file = os.path.join(report_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save trades to CSV
        trades_file = os.path.join(report_dir, "trades.csv")
        trades_data = [t.to_dict() for t in self.closed_trades]
        pd.DataFrame(trades_data).to_csv(trades_file, index=False)
        
        # Generate equity curve plot
        self._plot_equity_curve(report_dir)
        
        # Generate trade analysis plots
        self._plot_trade_analysis(report_dir)
        
        logger.info(f"Reports generated in: {report_dir}")
    
    def _plot_equity_curve(self, output_dir: str) -> None:
        """Generate equity curve plot."""
        if not self.equity_curve:
            return
            
        df = pd.DataFrame(self.equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['balance'], label='Equity Curve', linewidth=2)
        plt.title(f'Equity Curve - {self.config.symbol} ({self.config.timeframe})')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.legend()
        
        # Add drawdown areas
        peak = df['balance'].cummax()
        drawdown = (df['balance'] - peak) / peak
        plt.fill_between(df.index, df['balance'], peak, where=(df['balance'] < peak), 
                        color='red', alpha=0.3, label='Drawdown')
        
        # Save plot
        plot_file = os.path.join(output_dir, "equity_curve.png")
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_analysis(self, output_dir: str) -> None:
        """Generate trade analysis plots."""
        if not self.closed_trades:
            return
            
        df = pd.DataFrame([t.to_dict() for t in self.closed_trades])
        
        # P&L distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(df['pnl'], kde=True, bins=30)
        plt.title('P&L Distribution')
        plt.xlabel('P&L')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "pnl_distribution.png"), bbox_inches='tight')
        plt.close()
        
        # Win/Loss pie chart
        plt.figure(figsize=(6, 6))
        df['result'] = df['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
        df['result'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Win/Loss Ratio')
        plt.ylabel('')
        plt.savefig(os.path.join(output_dir, "win_loss_ratio.png"), bbox_inches='tight')
        plt.close()
    
    # Technical indicators
    def _calculate_rsi(self, candle: Dict[str, float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        # Simplified implementation - in a real scenario, use a proper RSI calculation
        # that considers previous candles
        return 50.0  # Placeholder
    
    def _calculate_macd(self, candle: Dict[str, float]) -> float:
        """Calculate MACD."""
        # Simplified implementation
        return 0.0  # Placeholder
    
    def _calculate_atr(self, candle: Dict[str, float], period: int = 14) -> float:
        """Calculate Average True Range."""
        # Simplified implementation
        return (candle['high'] - candle['low']) * 0.1  # Placeholder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest Orchestrator')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (e.g., 1h, 4h, 1d)')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0-1)')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    return parser.parse_args()

def main():
    """Main entry point for the backtest orchestrator."""
    args = parse_args()
    
    # Create backtest configuration
    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        output_dir=args.output
    )
    
    # Initialize and run backtest
    orchestrator = BacktestOrchestrator(config)
    metrics = orchestrator.run()
    
    # Print summary
    print("\n=== Backtest Summary ===")
    print(f"Symbol: {metrics['symbol']} ({metrics['timeframe']})")
    print(f"Period: {metrics['start_date']} to {metrics['end_date']}")
    print(f"Initial Balance: ${metrics['initial_balance']:,.2f}")
    print(f"Final Balance: ${metrics['final_balance']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print("=======================")

if __name__ == "__main__":
    main()