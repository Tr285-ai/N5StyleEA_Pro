"""
Reinforcement Learning Strategy for Trading Optimization

This module implements a Deep Q-Network (DQN) based trading strategy
that can learn and optimize trading decisions in real-time.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """Experience replay memory for DQN training."""
    
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self) -> int:
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network for trading strategy."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RLStrategy:
    """Reinforcement Learning based trading strategy."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001,
                 target_update: int = 100,
                 use_double_dqn: bool = True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.use_double_dqn = use_double_dqn
        self.steps_done = 0
        
        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Track performance
        self.episode_rewards = []
        self.episode_losses = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, next_state: np.ndarray, 
                reward: float, done: bool):
        """Store experience in replay memory."""
        self.memory.push(state, action, next_state, reward, done)
    
    def replay(self) -> Optional[float]:
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([s for s in batch.next_state if s is not None])).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=self.device, dtype=torch.bool)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        if self.use_double_dqn:
            # Double DQN
            next_state_actions = self.policy_net(next_state_batch).max(1)[1].detach()
            next_state_values[non_final_mask] = self.target_net(next_state_batch).gather(1, next_state_actions.unsqueeze(1)).squeeze(1).detach()
        else:
            # Standard DQN
            next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        
        return loss.item()
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.target_net.eval()

class TradingEnvironment:
    """Trading environment for RL training."""
    
    def __init__(self, 
                 data: np.ndarray, 
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 window_size: int = 10):
        """
        Initialize the trading environment.
        
        Args:
            data: Historical price data (OHLCV format)
            initial_balance: Initial account balance
            commission: Trading commission per trade
            window_size: Number of past time steps to include in state
        """
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        
        # Reset environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.done = False
        self.profits = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get the current state."""
        # Include OHLCV data for the last 'window_size' steps
        state = self.data[self.current_step - self.window_size:self.current_step]
        
        # Add position and balance information
        position_info = np.array([self.position, self.balance / self.initial_balance])
        
        # Flatten and normalize the state
        state = np.concatenate([state.flatten(), position_info])
        return (state - np.mean(state)) / (np.std(state) + 1e-8)  # Normalize
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action in the environment.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            
        Returns:
            next_state: Next state
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
        
        current_price = self.data[self.current_step, 3]  # Close price
        reward = 0.0
        
        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            if self.position < 0:
                # Close short position
                pnl = (self.entry_price - current_price) * abs(self.position)
                self.balance += pnl * (1 - self.commission)
                reward += pnl / self.initial_balance
                self.position = 0.0
            
            # Open long position
            position_size = self.balance * 0.99 / current_price  # Use 99% of balance
            self.position = position_size
            self.entry_price = current_price
            self.balance -= position_size * current_price * (1 + self.commission)
            
        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:
                # Close long position
                pnl = (current_price - self.entry_price) * self.position
                self.balance += pnl * (1 - self.commission)
                reward += pnl / self.initial_balance
                self.position = 0.0
            
            # Open short position
            position_size = self.balance * 0.99 / current_price  # Use 99% of balance
            self.position = -position_size
            self.entry_price = current_price
            self.balance -= position_size * current_price * (1 + self.commission)
        
        # Calculate unrealized P&L
        if self.position != 0:
            if self.position > 0:  # Long
                unrealized_pnl = (current_price - self.entry_price) * self.position
            else:  # Short
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
            
            # Add small reward for unrealized P&L
            reward += 0.1 * (unrealized_pnl / self.initial_balance)
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Calculate total portfolio value
        portfolio_value = self.balance
        if self.position > 0:  # Long
            portfolio_value += self.position * current_price
        elif self.position < 0:  # Short
            portfolio_value += self.position * current_price
        
        # Store profit for this step
        self.profits.append(portfolio_value - self.initial_balance)
        
        # Additional info
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'current_price': current_price,
            'step': self.current_step
        }
        
        return self._get_state(), reward, self.done, info

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_steps = 1000
    prices = np.cumprod(1 + np.random.normal(0.001, 0.02, n_steps)) * 100
    data = np.column_stack([
        prices * (1 + np.random.normal(0, 0.01, n_steps)),  # High
        prices * (1 + np.random.normal(0, 0.01, n_steps)),  # Low
        prices,                                             # Close
        np.random.normal(1000, 100, n_steps)                # Volume
    ])
    
    # Create environment
    env = TradingEnvironment(data)
    
    # Initialize RL strategy
    state_size = len(env.reset())
    action_size = 3  # Hold, Buy, Sell
    agent = RLStrategy(state_size, action_size)
    
    # Training loop
    n_episodes = 100
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            # Select action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, next_state, reward, done)
            
            # Train on batch
            loss = agent.replay()
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Portfolio Value: ${info['portfolio_value']:.2f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
                break
