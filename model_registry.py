from typing import List, Dict, Any, Optional
import joblib
import numpy as np
from tensorflow.keras.models import load_model, clone_model
import tensorflow as tf
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_type: str
    params: Dict[str, Any]
    weight: float = 1.0

class ModelEnsemble:
    def __init__(self, models_config: List[ModelConfig]):
        self.models = []
        self.weights = []
        
        for config in models_config:
            try:
                model = self._initialize_model(config)
                self.models.append(model)
                self.weights.append(config.weight)
                logger.info(f"Initialized {config.model_type} model")
            except Exception as e:
                logger.error(f"Failed to initialize {config.model_type}: {str(e)}")
                raise

        if not self.models:
            raise ValueError("No valid models were initialized")
            
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w/total_weight for w in self.weights]

    def _initialize_model(self, config: ModelConfig) -> tf.keras.Model:
        """Initialize a single model based on configuration"""
        if config.model_type == 'lstm':
            return self._create_lstm_model(config.params)
        elif config.model_type == 'cnn':
            return self._create_cnn_model(config.params)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def _create_lstm_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """Create LSTM model with given parameters"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=params.get('units', 64),
                input_shape=params['input_shape'],
                return_sequences=True
            ),
            tf.keras.layers.Dropout(params.get('dropout', 0.2)),
            tf.keras.layers.LSTM(units=params.get('units', 32)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _create_cnn_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """Create CNN model with given parameters"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                filters=params.get('filters', 64),
                kernel_size=params.get('kernel_size', 3),
                activation='relu',
                input_shape=params['input_shape']
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble"""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, verbose=0)
            predictions.append(pred * weight)
        
        # Weighted average of predictions
        return np.mean(predictions, axis=0)

    def save(self, filepath: str) -> None:
        """Save the ensemble to disk"""
        for i, model in enumerate(self.models):
            model.save(f"{filepath}_model_{i}.h5")
        joblib.dump(self.weights, f"{filepath}_weights.pkl")

    @classmethod
    def load(cls, filepath: str, models_config: List[ModelConfig]) -> 'ModelEnsemble':
        """Load a saved ensemble"""
        ensemble = cls(models_config)
        for i in range(len(ensemble.models)):
            ensemble.models[i] = load_model(f"{filepath}_model_{i}.h5")
        ensemble.weights = joblib.load(f"{filepath}_weights.pkl")
        return ensemble