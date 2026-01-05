import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TradingEnvironment:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.current_step = 0
        self.episode = 0
        
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode."""
        self.balance = self.initial_balance
        self.positions = {}
        self.current_step = 0
        self.episode += 1
        return self._get_state()
    
    def step(self, action: int, market_data: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            market_data: Current market data
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.current_step += 1
        current_price = market_data['close']
        reward = 0
        done = False
        
        # Execute action
        if action == 1:  # Buy
            position_size = min(self.balance * 0.1, self.balance / current_price)
            self.positions['long'] = {
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': current_price * 0.99,
                'take_profit': current_price * 1.02
            }
            self.balance -= position_size * current_price
            reward = -0.001  # Small negative reward for entering a trade
            
        elif action == 2:  # Sell
            position_size = min(self.balance * 0.1, self.balance / current_price)
            self.positions['short'] = {
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': current_price * 1.01,
                'take_profit': current_price * 0.98
            }
            self.balance += position_size * current_price
            reward = -0.001  # Small negative reward for entering a trade
            
        # Update existing positions
        portfolio_value = self.balance
        for pos_type, pos in list(self.positions.items()):
            if pos_type == 'long':
                pnl = (current_price - pos['entry_price']) * pos['size']
                if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                    self._close_position(pos_type, current_price)
                    reward = pnl / (pos['entry_price'] * pos['size'])  # ROI
            elif pos_type == 'short':
                pnl = (pos['entry_price'] - current_price) * pos['size']
                if current_price >= pos['stop_loss'] or current_price <= pos['take_profit']:
                    self._close_position(pos_type, current_price)
                    reward = pnl / (pos['entry_price'] * pos['size'])  # ROI
            portfolio_value += pnl
            
        # Update state
        next_state = self._get_state()
        done = self.current_step >= 1000  # End of episode
        
        # Log the trade
        self.trade_history.append({
            'episode': self.episode,
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'timestamp': datetime.now()
        })
        
        return next_state, reward, done, {'portfolio_value': portfolio_value}
    
    def _close_position(self, pos_type: str, exit_price: float):
        """Close a position and update balance."""
        pos = self.positions.pop(pos_type, None)
        if pos:
            if pos_type == 'long':
                self.balance += pos['size'] * exit_price
            else:  # short
                self.balance += pos['size'] * (2 * pos['entry_price'] - exit_price)
    
    def _get_state(self) -> np.ndarray:
        """Get the current state of the environment."""
        # This is a placeholder - you'll need to customize based on your needs
        return np.array([self.balance / self.initial_balance])  # Normalized balance

class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        memory_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model."""
        model = tf.keras.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the online model."""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray, is_training: bool = True) -> int:
        """Choose an action using epsilon-greedy policy."""
        if is_training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int) -> float:
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return 0.0
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        # Predict Q-values for current and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values using Bellman equation
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train the model
        history = self.model.fit(states, target, epochs=1, verbose=0)
        loss = history.history['loss'][0] if 'loss' in history.history else 0.0
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def save(self, path: str):
        """Save the model weights."""
        self.model.save_weights(path)
        
    def load(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            self.model.load_weights(path)
            self.update_target_model()
            return True
        return False

class RLTrader:
    def __init__(
        self,
        state_size: int = 10,
        action_size: int = 3,  # hold, buy, sell
        initial_balance: float = 10000.0,
        model_path: Optional[str] = None
    ):
        self.env = TradingEnvironment(initial_balance)
        self.state_size = state_size
        self.action_size = action_size
        self.agent = DQNAgent(state_size, action_size)
        self.model_path = model_path or 'models/rl_trader.h5'
        
        # Load existing model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, market_data: pd.DataFrame, episodes: int = 100, batch_size: int = 32):
        """Train the RL agent."""
        total_rewards = []
        
        for e in range(episodes):
            state = self.env.reset()
            state = self._process_state(state, market_data.iloc[0])
            total_reward = 0
            done = False
            i = 0
            
            while not done and i < len(market_data) - 1:
                # Get action from agent
                action = self.agent.act(state)
                
                # Take action and get next state
                next_state, reward, done, _ = self.env.step(action, market_data.iloc[i].to_dict())
                next_state = self._process_state(next_state, market_data.iloc[i+1])
                
                # Store experience in replay memory
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train the agent
                if len(self.agent.memory) > batch_size:
                    loss = self.agent.replay(batch_size)
                
                total_reward += reward
                state = next_state
                i += 1
                
            # Update target network
            self.agent.update_target_model()
            
            # Save model periodically
            if e % 10 == 0:
                self.save_model(self.model_path)
                
            total_rewards.append(total_reward)
            logger.info(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        return total_rewards
    
    def predict(self, state: np.ndarray, market_data: pd.Series) -> Tuple[int, float]:
        """Predict the best action for the current state."""
        state = self._process_state(state, market_data)
        q_values = self.agent.model.predict(np.reshape(state, [1, self.state_size]), verbose=0)[0]
        action = np.argmax(q_values)
        confidence = float(q_values[action])
        return action, confidence
    
    def _process_state(self, state: np.ndarray, market_data: pd.Series) -> np.ndarray:
        """Process the state with market data."""
        # Customize this based on your state representation
        features = np.array([
            market_data['close'],
            market_data['volume'],
            market_data.get('rsi', 50) / 100,  # Normalize RSI to [0,1]
            market_data.get('macd', 0),
            market_data.get('bb_upper', 0),
            market_data.get('bb_middle', 0),
            market_data.get('bb_lower', 0),
            market_data['close'] / market_data['open'] - 1,  # Price change
            market_data['high'] - market_data['low'],  # Volatility
            state[0]  # Current portfolio value
        ])
        return features
    
    def save_model(self, path: str):
        """Save the RL model."""
        self.agent.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load a trained RL model."""
        if os.path.exists(path):
            self.agent.load(path)
            logger.info(f"Model loaded from {path}")
            return True
        logger.warning(f"No model found at {path}")
        return False