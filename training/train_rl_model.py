"""
Reinforcement Learning Model Training

This script trains a reinforcement learning model using historical market data.
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ai.rl_strategy import RLStrategy, TradingEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLModelTrainer:
    """Handles training of reinforcement learning models for trading."""
    
    def __init__(self, config_path: str = None):
        """Initialize the RL trainer with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.env = None
        
        # Set random seeds for reproducibility
        self._set_seeds(self.config.get('random_seed', 42))
        
        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'models/rl_models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            # Data settings
            'data_path': 'data/ohlcv_data.csv',
            'train_test_split': 0.8,
            'window_size': 20,
            
            # Environment settings
            'initial_balance': 10000.0,
            'commission': 0.001,
            'max_steps': 1000,
            
            # Model settings
            'state_size': 20 * 5 + 2,  # window_size * features + position info
            'action_size': 3,  # hold, buy, sell
            'hidden_size': 128,
            'batch_size': 64,
            'memory_size': 10000,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'learning_rate': 0.001,
            'target_update': 100,
            'use_double_dqn': True,
            
            # Training settings
            'num_episodes': 1000,
            'save_interval': 100,
            'eval_interval': 10,
            'checkpoint_dir': 'checkpoints',
            'tensorboard_logdir': 'runs',
            
            # Hyperparameter search
            'hyperparameter_search': False,
            'hyperparameters': {
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64, 128],
                'hidden_size': [64, 128, 256]
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using default configuration.")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using default configuration.")
            return default_config
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess market data."""
        logger.info(f"Loading data from {self.config['data_path']}")
        
        try:
            # Load OHLCV data
            df = pd.read_csv(self.config['data_path'])
            
            # Convert to numpy array
            # Assuming columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            data = df[['open', 'high', 'low', 'close', 'volume']].values
            
            # Normalize the data
            data_mean = data.mean(axis=0)
            data_std = data.std(axis=0) + 1e-8  # Avoid division by zero
            data = (data - data_mean) / data_std
            
            # Split into train and test sets
            split_idx = int(len(data) * self.config['train_test_split'])
            train_data = data[:split_idx]
            test_data = data[split_idx:]
            
            logger.info(f"Data loaded: {len(train_data)} training samples, {len(test_data)} test samples")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_environment(self, data: np.ndarray) -> TradingEnvironment:
        """Create a trading environment with the given data."""
        return TradingEnvironment(
            data=data,
            initial_balance=self.config['initial_balance'],
            commission=self.config['commission'],
            window_size=self.config['window_size']
        )
    
    def create_model(self) -> RLStrategy:
        """Create a new RL model."""
        return RLStrategy(
            state_size=self.config['state_size'],
            action_size=self.config['action_size'],
            device=self.device,
            memory_size=self.config['memory_size'],
            batch_size=self.config['batch_size'],
            gamma=self.config['gamma'],
            epsilon=self.config['epsilon'],
            epsilon_min=self.config['epsilon_min'],
            epsilon_decay=self.config['epsilon_decay'],
            learning_rate=self.config['learning_rate'],
            target_update=self.config['target_update'],
            use_double_dqn=self.config['use_double_dqn']
        )
    
    def train(self):
        """Train the RL model."""
        logger.info("Starting RL model training...")
        
        # Load data
        train_data, test_data = self.load_data()
        
        # Create environments
        train_env = self.create_environment(train_data)
        test_env = self.create_environment(test_data)
        
        # Initialize model
        self.model = self.create_model()
        
        # Training loop
        best_reward = -float('inf')
        
        for episode in range(1, self.config['num_episodes'] + 1):
            # Train episode
            train_reward = self._run_episode(train_env, training=True)
            
            # Evaluate periodically
            if episode % self.config['eval_interval'] == 0:
                test_reward = self._run_episode(test_env, training=False)
                logger.info(
                    f"Episode {episode}/{self.config['num_episodes']} | "
                    f"Train Reward: {train_reward:.2f} | "
                    f"Test Reward: {test_reward:.2f} | "
                    f"Epsilon: {self.model.epsilon:.4f}"
                )
                
                # Save best model
                if test_reward > best_reward:
                    best_reward = test_reward
                    self.save_model(f"best_model_ep{episode}_reward{test_reward:.2f}.pth")
            
            # Save checkpoint
            if episode % self.config['save_interval'] == 0:
                self.save_model(f"checkpoint_ep{episode}.pth")
        
        logger.info("Training completed!")
    
    def _run_episode(self, env: TradingEnvironment, training: bool = True) -> float:
        """Run a single episode and return the total reward."""
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            # Select action
            action = self.model.get_action(state, training=training)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            if training:
                self.model.remember(state, action, next_state, reward, done)
                
                # Train on batch
                if len(self.model.memory) > self.config['batch_size']:
                    self.model.replay()
            
            state = next_state
            total_reward += reward
        
        return total_reward
    
    def save_model(self, filename: str):
        """Save the model to disk."""
        if not self.model:
            logger.warning("No model to save")
            return
            
        path = self.output_dir / filename
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def hyperparameter_search(self):
        """Perform hyperparameter search."""
        if not self.config['hyperparameter_search']:
            logger.info("Hyperparameter search is disabled")
            return
        
        logger.info("Starting hyperparameter search...")
        
        # Load data
        train_data, test_data = self.load_data()
        train_env = self.create_environment(train_data)
        
        # Grid search
        best_params = None
        best_reward = -float('inf')
        
        # Generate all combinations
        from itertools import product
        param_grid = self.config['hyperparameters']
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for values in product(*param_values):
            # Update config with current hyperparameters
            params = dict(zip(param_names, values))
            logger.info(f"Testing hyperparameters: {params}")
            
            # Update model config
            for key, value in params.items():
                if key in self.config:
                    self.config[key] = value
            
            # Create and train model
            self.model = self.create_model()
            reward = self._run_episode(train_env, training=True)
            
            logger.info(f"Hyperparameters: {params} | Reward: {reward:.2f}")
            
            # Update best parameters
            if reward > best_reward:
                best_reward = reward
                best_params = params
                logger.info(f"New best parameters found: {best_params} (reward: {best_reward:.2f})")
        
        logger.info(f"Best hyperparameters: {best_params} (reward: {best_reward:.2f})")
        return best_params

def main():
    """Main function for training RL model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL trading model')
    parser.add_argument('--config', type=str, default='config/rl_config.json',
                      help='Path to configuration file')
    parser.add_argument('--hyperparameter-search', action='store_true',
                      help='Perform hyperparameter search')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RLModelTrainer(args.config)
    
    # Run hyperparameter search if requested
    if args.hyperparameter_search:
        trainer.config['hyperparameter_search'] = True
        trainer.hyperparameter_search()
    else:
        # Train model
        trainer.train()

if __name__ == "__main__":
    main()
