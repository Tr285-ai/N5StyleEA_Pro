# evolving_trader_combined.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import random
from dataclasses import dataclass
import logging
from collections import deque
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    entry_price: float
    exit_price: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    pnl: float = 0.0
    status: str = "open"  # open, closed, stopped_out

class EvolvingTrader:
    def __init__(self, initial_balance: float = 10000.0, risk_per_trade: float = 0.01):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.positions: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.learning_rate = 0.01
        self.memory = deque(maxlen=1000)
        self.state_size = 10  # Number of features in state
        self.action_size = 3  # Buy, Sell, Hold
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a simple neural network model for Q-learning."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def get_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Convert market data into a state representation."""
        # This is a simplified example - replace with your actual feature engineering
        return np.array([
            market_data.get('rsi', 50) / 100,  # Normalize to 0-1
            market_data.get('macd', 0) / 100,  # Normalize
            market_data.get('bb_upper', 0) / market_data.get('price', 1),
            market_data.get('bb_lower', 0) / market_data.get('price', 1),
            market_data.get('volume', 0) / 1e6,  # Normalize volume
            market_data.get('price', 0) / market_data.get('sma_50', 1),
            market_data.get('price', 0) / market_data.get('sma_200', 1),
            market_data.get('atr', 0) / market_data.get('price', 1),
            market_data.get('adx', 0) / 100,  # Normalize ADX
            len(self.positions) / 10  # Position count normalized
        ])
    
    def act(self, state: np.ndarray) -> int:
        """Select an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """Train the model on past experiences."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        targets = self.model.predict(states)
        next_q_values = self.model.predict(next_states)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def execute_trade(self, symbol: str, action: int, price: float, timestamp: str) -> bool:
        """Execute a trade based on the action."""
        action_map = {0: 'buy', 1: 'sell', 2: 'hold'}
        action_str = action_map.get(action, 'hold')
        
        if action_str == 'hold':
            return False
            
        if action_str == 'buy' and symbol not in self.positions:
            # Calculate position size based on risk
            risk_amount = self.balance * self.risk_per_trade
            position_size = risk_amount / (price * 0.02)  # 2% stop loss
            
            self.positions[symbol] = Trade(
                symbol=symbol,
                entry_price=price,
                entry_time=timestamp
            )
            self.balance -= position_size * price
            return True
            
        elif action_str == 'sell' and symbol in self.positions:
            trade = self.positions[symbol]
            trade.exit_price = price
            trade.exit_time = timestamp
            trade.pnl = (price - trade.entry_price) * (self.balance * self.risk_per_trade / trade.entry_price)
            trade.status = 'closed'
            
            self.balance += (self.balance * self.risk_per_trade) + trade.pnl
            self.trade_history.append(trade)
            del self.positions[symbol]
            return True
            
        return False
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            (trade.entry_price * (self.balance * self.risk_per_trade / trade.entry_price))
            for trade in self.positions.values()
        )
        return self.balance + positions_value
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.trade_history:
            return {}
            
        returns = np.array([t.pnl for t in self.trade_history])
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        
        return {
            'total_return': self.get_portfolio_value() / self.initial_balance - 1,
            'total_trades': len(self.trade_history),
            'win_rate': len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in self.trade_history if t.pnl <= 0]) if any(t.pnl <= 0 for t in self.trade_history) else 0,
            'profit_factor': abs(sum(t.pnl for t in self.trade_history if t.pnl > 0) / 
                              sum(abs(t.pnl) for t in self.trade_history if t.pnl < 0)) if any(t.pnl < 0 for t in self.trade_history) else float('inf')
        }