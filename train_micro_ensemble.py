"""
Train Micro Ensemble v15.2

A streamlined script for training an ensemble of micro-prediction models
for financial time series forecasting. Supports LSTM, CNN, and Transformer models.
"""

# Core Imports
import os
import json
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class ModelType(Enum):
    LSTM = auto()
    CNN = auto()
    TRANSFORMER = auto()

@dataclass
class ModelConfig:
    model_type: ModelType
    params: Dict
    input_shape: Tuple[int, ...]
    num_classes: int
    name: str = ""

class MicroEnsemble:
    def __init__(self, model_configs: List[ModelConfig], output_dir: str = 'models'):
        self.model_configs = model_configs
        self.output_dir = Path(output_dir)
        self.models = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_models(self):
        """Initialize all models in the ensemble."""
        for config in self.model_configs:
            if config.model_type == ModelType.LSTM:
                model = self._build_lstm(config)
            elif config.model_type == ModelType.CNN:
                model = self._build_cnn(config)
            elif config.model_type == ModelType.TRANSFORMER:
                model = self._build_transformer(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            self.models.append({
                'config': config,
                'model': model
            })

    def _build_lstm(self, config: ModelConfig) -> tf.keras.Model:
        """Build an LSTM model."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, 
                               input_shape=config.input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(config.num_classes, activation='softmax')
        ])
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train all models in the ensemble."""
        for i, model_info in enumerate(self.models):
            model = model_info['model']
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Save the trained model
            model_path = self.output_dir / f"{model_info['config'].name or f'model_{i}'}.h5"
            model.save(model_path)

def example_usage():
    # Example configuration
    model_configs = [
        ModelConfig(
            model_type=ModelType.LSTM,
            params={},
            input_shape=(60, 10),  # 60 time steps, 10 features
            num_classes=3,         # 3 output classes
            name="lstm_model"
        )
    ]
    
    # Initialize and train ensemble
    ensemble = MicroEnsemble(model_configs)
    ensemble.build_models()
    
    # Generate sample data
    X_train = np.random.randn(1000, 60, 10)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 3, 1000), 3)
    
    # Train the ensemble
    ensemble.train(X_train, y_train, epochs=10, batch_size=32)

if __name__ == "__main__":
    example_usage()