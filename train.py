import pandas as pd
import numpy as np
from model_registry import ModelEnsemble
import joblib
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def prepare_training_data(data: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for the model.
    
    Args:
        data: DataFrame containing OHLCV data
        window_size: Number of time steps to use as input
        
    Returns:
        Tuple of (X, y) where X is the input features and y is the target
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        # Create input sequence
        X.append(data.iloc[i-window_size:i][['open', 'high', 'low', 'close', 'volume']].values)
        # Create target (1 if price went up, 0 if down)
        y.append(1 if data['close'].iloc[i] > data['close'].iloc[i-1] else 0)
    return np.array(X), np.array(y)

def train_ensemble(data: pd.DataFrame, window_size: int = 30, save_path: str = "models/ensemble") -> ModelEnsemble:
    """
    Train the model ensemble and save it to disk.
    
    Args:
        data: Training data
        window_size: Number of time steps to use as input
        save_path: Directory to save the trained model
        
    Returns:
        Trained ModelEnsemble instance
    """
    # Prepare data
    X, y = prepare_training_data(data, window_size)
    
    # Initialize and train ensemble
    input_shape = (window_size, X.shape[2])  # (time_steps, features)
    ensemble = ModelEnsemble(input_shape=input_shape)
    ensemble.train(X, y)
    
    # Save the ensemble
    ensemble.save(save_path)
    logger.info(f"Model ensemble saved to {save_path}")
    return ensemble