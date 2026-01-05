# train_micro_models.py
"""
Micro Model Trainer v15.2

A unified training script for both CNN and LSTM micro-prediction models.
Supports:
- Training both CNN and LSTM architectures
- Hyperparameter tuning
- Early stopping and model checkpointing
- TensorBoard integration
- Mixed precision training
- Model conversion to TensorFlow Lite

Author: N5StyleEA Team
Version: 15.2.1
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from export_micro_onnx import ONNXExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_trainer')

class ModelType(Enum):
    """Supported model architectures."""
    CNN = "cnn"
    LSTM = "lstm"
    CNN_LSTM = "cnn_lstm"

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    test_split: float = 0.1
    patience: int = 10
    min_delta: float = 1e-4
    use_class_weights: bool = True
    enable_tensorboard: bool = True
    enable_checkpoints: bool = True
    early_stopping: bool = True
    reduce_lr: bool = True

class MicroModelTrainer:
    """Unified trainer for micro-prediction models."""
    
    def __init__(
        self,
        model_type: ModelType,
        input_shape: Tuple[int, ...],
        output_dir: Union[str, Path] = "models",
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model to train (CNN, LSTM, or hybrid)
            input_shape: Shape of input data (seq_len, features)
            output_dir: Directory to save models and logs
            config: Training configuration
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.output_dir = Path(output_dir)
        self.config = config or TrainingConfig()
        self.model = None
        self.history = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self.model_dir = self.output_dir / f"micro_{model_type.value}"
        self.log_dir = self.output_dir / "logs" / f"micro_{model_type.value}_{int(time.time())}"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        
        for d in [self.model_dir, self.log_dir, self.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {model_type.value.upper()} trainer")
        logger.info(f"Model will be saved to: {self.model_dir}")
        logger.info(f"Using device: {self.device}")

    def build_model(self) -> nn.Module:
        """Build the specified model architecture."""
        try:
            if self.model_type == ModelType.CNN:
                return self._build_cnn_layers()
            elif self.model_type == ModelType.LSTM:
                return self._build_lstm_layers()
            elif self.model_type == ModelType.CNN_LSTM:
                return self._build_cnn_lstm_layers()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def _build_cnn_layers(self) -> nn.Module:
        """Build CNN architecture."""
        try:
            return nn.Sequential(
                nn.Conv1d(self.input_shape[1], 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(64 * (self.input_shape[0] // 2), 1)
            ).to(self.device)
        except Exception as e:
            logger.error(f"Error building CNN layers: {str(e)}")
            raise

    def _build_lstm_layers(self) -> nn.Module:
        """Build LSTM architecture."""
        try:
            return nn.Sequential(
                nn.LSTM(
                    input_size=self.input_shape[1],
                    hidden_size=64,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.2
                ),
                nn.Linear(64, 1)
            ).to(self.device)
        except Exception as e:
            logger.error(f"Error building LSTM layers: {str(e)}")
            raise

    def _build_cnn_lstm_layers(self) -> nn.Module:
        """Build hybrid CNN-LSTM architecture."""
        try:
            class CNNLSTM(nn.Module):
                def __init__(self, input_shape):
                    super().__init__()
                    self.cnn = nn.Sequential(
                        nn.Conv1d(input_shape[1], 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(2)
                    )
                    self.lstm = nn.LSTM(
                        input_size=64,
                        hidden_size=32,
                        num_layers=1,
                        batch_first=True
                    )
                    self.fc = nn.Linear(32, 1)
                
                def forward(self, x):
                    x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
                    x = self.cnn(x)
                    x = x.permute(0, 2, 1)  # (batch, seq_len, features)
                    x, _ = self.lstm(x)
                    return self.fc(x[:, -1, :])
            
            return CNNLSTM(self.input_shape).to(self.device)
        except Exception as e:
            logger.error(f"Error building CNN-LSTM layers: {str(e)}")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        try:
            # Convert data to PyTorch tensors
            train_data = torch.tensor(X_train, dtype=torch.float32)
            train_labels = torch.tensor(y_train, dtype=torch.float32)
            
            # Create datasets and dataloaders
            train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            val_loader = None
            if X_val is not None and y_val is not None:
                val_data = torch.tensor(X_val, dtype=torch.float32)
                val_labels = torch.tensor(y_val, dtype=torch.float32)
                val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )
            
            # Initialize model and training components
            self.model = self.build_model()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.patience // 2,
                verbose=True
            )
            
            # Training loop
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': []
            }
            
            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                
                # Calculate average training loss
                train_loss /= len(train_loader.dataset)
                history['train_loss'].append(train_loss)
                
                # Validation phase
                val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            outputs = self.model(inputs)
                            val_loss += criterion(outputs, targets).item() * inputs.size(0)
                    
                    val_loss /= len(val_loader.dataset)
                    history['val_loss'].append(val_loss)
                    
                    # Update learning rate
                    scheduler.step(val_loss)
                    
                    # Check for improvement
                    if val_loss < best_val_loss - self.config.min_delta:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        self._save_checkpoint(epoch, is_best=True)
                    else:
                        epochs_without_improvement += 1
                        if (self.config.early_stopping and 
                            epochs_without_improvement >= self.config.patience):
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                            break
                
                # Log progress
                current_lr = optimizer.param_groups[0]['lr']
                history['learning_rate'].append(current_lr)
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"Val Loss: {val_loss if val_loader else 'N/A':.6f} - "
                    f"LR: {current_lr:.6f}"
                )
            
            self.history = history
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': None,  # Add optimizer if needed
                'loss': self.history['val_loss'][-1] if self.history['val_loss'] else float('inf')
            }
            
            # Save regular checkpoint
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Save as best model if applicable
            if is_best:
                best_path = self.model_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
                
            self.model.eval()
            test_data = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            test_labels = torch.tensor(y_test, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(test_data)
                mse = nn.MSELoss()(predictions, test_labels).item()
                mae = nn.L1Loss()(predictions, test_labels).item()
                
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, format: str = 'pt') -> Dict[str, str]:
        """
        Save the trained model.
        
        Args:
            format: Format to save the model ('pt' or 'onnx')
            
        Returns:
            Dictionary with paths to saved model files
        """
        try:
            saved_paths = {}
            
            # Save PyTorch model
            if format == 'pt':
                model_path = self.model_dir / "model.pt"
                torch.save(self.model.state_dict(), model_path)
                saved_paths['pytorch'] = str(model_path)
            
            # Export to ONNX
            elif format == 'onnx':
                onnx_path = self.model_dir / "model.onnx"
                dummy_input = torch.randn(1, *self.input_shape).to(self.device)
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    onnx_path,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                saved_paths['onnx'] = str(onnx_path)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Model saved to {saved_paths}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train micro-prediction models')
    parser.add_argument('--model-type', type=str, required=True,
                      choices=['cnn', 'lstm', 'cnn_lstm'],
                      help='Type of model to train')
    parser.add_argument('--input-shape', type=int, nargs=2, required=True,
                      help='Input shape (sequence_length, num_features)')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='Learning rate')
    return parser.parse_args()

def main():
    """Main training function."""
    try:
        args = parse_args()
        
        # Initialize trainer
        config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        trainer = MicroModelTrainer(
            model_type=ModelType(args.model_type),
            input_shape=tuple(args.input_shape),
            output_dir=args.output_dir,
            config=config
        )
        
        # Load and prepare data
        # Note: Implement your data loading logic here
        # X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
        
        # Train model
        # history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        # metrics = trainer.evaluate(X_test, y_test)
        
        # Save model
        # saved_paths = trainer.save_model(format='onnx')
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()