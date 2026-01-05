"""
Micro Predictor v15.2

A high-performance prediction module for real-time market micro-predictions.
Combines multiple timeframes and technical indicators to generate short-term
price movement predictions with confidence scores.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import joblib
import talib
import logging
from pathlib import Path
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('micro_predictor.log')
    ]
)
logger = logging.getLogger('micro_predictor')

# Type aliases
Array = np.ndarray
DataFrame = pd.DataFrame
Tensor = tf.Tensor
Model = tf.keras.Model

class TimeFrame(Enum):
    """Supported timeframes for prediction."""
    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'

class PredictionType(Enum):
    """Types of predictions supported."""
    DIRECTION = 'direction'  # Up/Down prediction
    STRENGTH = 'strength'    # Strength of move
    VOLATILITY = 'volatility' # Expected volatility
    REVERSAL = 'reversal'    # Potential reversal points

@dataclass
class ModelConfig:
    """Configuration for a prediction model."""
    model_type: str
    input_shape: Tuple[int, ...]
    output_shape: int
    timeframes: List[TimeFrame]
    indicators: List[str]
    sequence_length: int = 60
    use_attention: bool = False
    dropout_rate: float = 0.2
    learning_rate: float = 1e-4
    name: str = ""

@dataclass
class PredictionResult:
    """Container for prediction results."""
    prediction: Array
    confidence: float
    model_name: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MicroPredictor:
    """
    Micro-prediction engine for real-time market analysis.
    
    This class handles multiple prediction models across different timeframes
    and combines their outputs for robust short-term market predictions.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        models_dir: str = "models/micro_predictors",
        warmup_period: int = 1000
    ):
        """
        Initialize the micro-predictor.
        
        Args:
            config_path: Path to configuration file
            models_dir: Directory containing model files
            warmup_period: Number of initial observations to collect before predicting
        """
        self.models: Dict[str, Model] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.models_dir = Path(models_dir)
        self.warmup_period = warmup_period
        self.data_buffer: Dict[TimeFrame, deque] = {
            tf: deque(maxlen=warmup_period) for tf in TimeFrame
        }
        self.last_prediction: Optional[PredictionResult] = None
        self.prediction_history: deque = deque(maxlen=1000)
        self.is_initialized = False
        self.lock = threading.RLock()
        
        # Initialize from config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # Start background thread for model updates
        self._stop_event = threading.Event()
        self._update_thread = threading.Thread(
            target=self._model_update_loop,
            daemon=True
        )
        self._update_thread.start()
        
        logger.info("MicroPredictor initialized")
    
    def __del__(self):
        """Cleanup resources."""
        self._stop_event.set()
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=5.0)
    
    def load_config(self, config_path: str) -> bool:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Clear existing models
            self.models.clear()
            self.model_configs.clear()
            
            # Load model configurations
            for model_cfg in config_data.get('models', []):
                cfg = ModelConfig(
                    model_type=model_cfg['model_type'],
                    input_shape=tuple(model_cfg['input_shape']),
                    output_shape=model_cfg['output_shape'],
                    timeframes=[TimeFrame(tf) for tf in model_cfg['timeframes']],
                    indicators=model_cfg.get('indicators', []),
                    sequence_length=model_cfg.get('sequence_length', 60),
                    use_attention=model_cfg.get('use_attention', False),
                    dropout_rate=model_cfg.get('dropout_rate', 0.2),
                    learning_rate=model_cfg.get('learning_rate', 1e-4),
                    name=model_cfg.get('name', '')
                )
                self.model_configs[cfg.name] = cfg
            
            logger.info(f"Loaded configuration from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def load_models(self) -> bool:
        """Load all configured models."""
        if not self.model_configs:
            logger.warning("No model configurations available")
            return False
        
        success = True
        for name, config in self.model_configs.items():
            model_path = self.models_dir / f"{name}.h5"
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                success = False
                continue
            
            try:
                # Load the model
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=self._get_custom_objects()
                )
                self.models[name] = model
                logger.info(f"Loaded model: {name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                success = False
        
        self.is_initialized = success
        return success
    
    def _get_custom_objects(self) -> Dict:
        """Get custom objects for model loading."""
        return {
            # Add any custom layers or metrics here
        }
    
    def add_market_data(
        self,
        timeframe: Union[str, TimeFrame],
        ohlcv: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add new market data to the predictor.
        
        Args:
            timeframe: Timeframe of the data (e.g., '1m', '5m')
            ohlcv: Dictionary with 'open', 'high', 'low', 'close', 'volume'
            timestamp: Optional timestamp (defaults to current time)
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
        
        if timestamp is None:
            timestamp = time.time()
        
        data_point = {
            'timestamp': timestamp,
            'open': float(ohlcv['open']),
            'high': float(ohlcv['high']),
            'low': float(ohlcv['low']),
            'close': float(ohlcv['close']),
            'volume': float(ohlcv.get('volume', 0))
        }
        
        with self.lock:
            self.data_buffer[timeframe].append(data_point)
    
    def _preprocess_data(
        self,
        data: List[Dict[str, float]],
        indicators: List[str]
    ) -> Array:
        """Preprocess data and calculate technical indicators."""
        if not data:
            raise ValueError("No data provided for preprocessing")
        
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        for indicator in indicators:
            if indicator == 'rsi':
                df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            elif indicator == 'macd':
                macd, signal, _ = talib.MACD(df['close'])
                df['macd'] = macd
                df['macd_signal'] = signal
            elif indicator == 'bollinger':
                upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
            # Add more indicators as needed
        
        # Normalize the data
        df_norm = (df - df.mean()) / (df.std() + 1e-8)
        return df_norm.values
    
    def predict(
        self,
        prediction_type: Union[str, PredictionType] = PredictionType.DIRECTION,
        timeframes: Optional[List[Union[str, TimeFrame]]] = None,
        models: Optional[List[str]] = None,
        ensemble_method: str = 'average'
    ) -> Dict[str, Any]:
        """
        Generate predictions using the loaded models.
        
        Args:
            prediction_type: Type of prediction to make
            timeframes: List of timeframes to use (None for all available)
            models: List of model names to use (None for all loaded)
            ensemble_method: How to combine predictions ('average', 'vote', 'best')
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_initialized:
            self.load_models()
            if not self.is_initialized:
                raise RuntimeError("Predictor not initialized")
        
        if isinstance(prediction_type, str):
            prediction_type = PredictionType(prediction_type.lower())
        
        if timeframes is None:
            timeframes = list(TimeFrame)
        else:
            timeframes = [TimeFrame(tf) if isinstance(tf, str) else tf 
                         for tf in timeframes]
        
        if models is None:
            models = list(self.models.keys())
        
        # Check if we have enough data
        for tf in timeframes:
            if len(self.data_buffer[tf]) < self.warmup_period:
                raise ValueError(f"Not enough data for {tf.value} timeframe")
        
        # Generate predictions
        predictions = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for model_name in models:
                if model_name not in self.models:
                    logger.warning(f"Model not found: {model_name}")
                    continue
                
                for tf in timeframes:
                    if tf not in self.model_configs[model_name].timeframes:
                        continue
                    
                    future = executor.submit(
                        self._predict_single,
                        model_name=model_name,
                        timeframe=tf,
                        prediction_type=prediction_type
                    )
                    futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        predictions.append(result)
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
        
        if not predictions:
            raise RuntimeError("No predictions were generated")
        
        # Combine predictions
        combined = self._combine_predictions(
            predictions,
            method=ensemble_method
        )
        
        # Store the last prediction
        self.last_prediction = PredictionResult(
            prediction=combined['prediction'],
            confidence=combined['confidence'],
            model_name='ensemble',
            timestamp=time.time(),
            metadata={
                'prediction_type': prediction_type.value,
                'timeframes': [tf.value for tf in timeframes],
                'models': models,
                'ensemble_method': ensemble_method
            }
        )
        
        self.prediction_history.append(self.last_prediction)
        return combined
    
    def _predict_single(
        self,
        model_name: str,
        timeframe: TimeFrame,
        prediction_type: PredictionType
    ) -> Optional[PredictionResult]:
        """Generate prediction using a single model and timeframe."""
        try:
            model = self.models[model_name]
            config = self.model_configs[model_name]
            
            # Get and preprocess data
            with self.lock:
                data = list(self.data_buffer[timeframe])
            
            # Ensure we have enough data
            if len(data) < config.sequence_length:
                logger.warning(f"Not enough data for {model_name} on {timeframe.value}")
                return None
            
            # Prepare input
            input_data = self._prepare_model_input(data, config)
            
            # Make prediction
            prediction = model.predict(
                input_data[np.newaxis, ...],
                verbose=0
            )[0]
            
            # Calculate confidence (simple max probability for classification)
            confidence = float(np.max(prediction))
            
            return PredictionResult(
                prediction=prediction,
                confidence=confidence,
                model_name=f"{model_name}_{timeframe.value}",
                timestamp=time.time(),
                metadata={
                    'timeframe': timeframe.value,
                    'prediction_type': prediction_type.value
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name} on {timeframe.value}: {e}")
            return None
    
    def _prepare_model_input(
        self,
        data: List[Dict[str, float]],
        config: ModelConfig
    ) -> Array:
        """Prepare input data for model prediction."""
        # Convert to DataFrame
        df = pd.DataFrame(data[-config.sequence_length:])
        
        # Calculate features
        features = []
        
        # Add OHLCV
        features.extend([
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values
        ])
        
        # Add technical indicators
        for indicator in config.indicators:
            if indicator == 'rsi':
                rsi = talib.RSI(df['close'], timeperiod=14).values
                features.append(rsi)
            elif indicator == 'macd':
                macd, signal, _ = talib.MACD(df['close'])
                features.extend([macd.values, signal.values])
            # Add more indicators as needed
        
        # Stack features and ensure correct shape
        features = np.stack(features, axis=-1)
        
        # Handle sequence length
        if len(features) > config.sequence_length:
            features = features[-config.sequence_length:]
        elif len(features) < config.sequence_length:
            # Pad with zeros if needed
            pad_width = [(0, config.sequence_length - len(features))] + [(0, 0)] * (features.ndim - 1)
            features = np.pad(features, pad_width, mode='constant')
        
        return features.astype(np.float32)
    
    def _combine_predictions(
        self,
        predictions: List[PredictionResult],
        method: str = 'average'
    ) -> Dict[str, Any]:
        """Combine multiple predictions into a single result."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        if method == 'average':
            # Simple average of predictions
            avg_pred = np.mean(
                [p.prediction for p in predictions],
                axis=0
            )
            avg_conf = np.mean([p.confidence for p in predictions])
            
            return {
                'prediction': avg_pred,
                'confidence': float(avg_conf),
                'num_predictions': len(predictions)
            }
            
        elif method == 'vote':
            # Majority vote for classification
            if len(predictions[0].prediction.shape) == 1:
                # Binary classification
                votes = np.round([p.prediction[0] for p in predictions])
                final_vote = np.round(np.mean(votes))
                confidence = float(np.mean([
                    p.confidence if np.round(p.prediction[0]) == final_vote 
                    else 1 - p.confidence 
                    for p in predictions
                ]))
                
                return {
                    'prediction': np.array([final_vote]),
                    'confidence': confidence,
                    'num_predictions': len(predictions)
                }
            else:
                # Multi-class classification
                class_votes = np.argmax(
                    np.stack([p.prediction for p in predictions]),
                    axis=1
                )
                from scipy.stats import mode
                final_vote = mode(class_votes).mode[0]
                confidence = float(np.mean([
                    p.confidence if np.argmax(p.prediction) == final_vote
                    else 1 - p.confidence
                    for p in predictions
                ]))
                
                # Create one-hot encoded prediction
                pred = np.zeros_like(predictions[0].prediction)
                pred[final_vote] = 1.0
                
                return {
                    'prediction': pred,
                    'confidence': confidence,
                    'num_predictions': len(predictions)
                }
        
        elif method == 'best':
            # Use prediction with highest confidence
            best_pred = max(predictions, key=lambda x: x.confidence)
            return {
                'prediction': best_pred.prediction,
                'confidence': best_pred.confidence,
                'model': best_pred.model_name,
                'num_predictions': len(predictions)
            }
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _model_update_loop(self) -> None:
        """Background thread for model updates and maintenance."""
        while not self._stop_event.is_set():
            try:
                # Check for model updates
                self._check_for_updates()
                
                # Clean up old predictions
                current_time = time.time()
                with self.lock:
                    self.prediction_history = deque(
                        p for p in self.prediction_history
                        if current_time - p.timestamp < 86400  # Keep 24h of history
                    )
                
                # Sleep for a while
                self._stop_event.wait(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _check_for_updates(self) -> None:
        """Check for and load updated models."""
        try:
            for model_name in list(self.models.keys()):
                model_path = self.models_dir / f"{model_name}.h5"
                if not model_path.exists():
                    continue
                
                # Check if file has been modified since last load
                mtime = os.path.getmtime(model_path)
                if hasattr(self, f"_last_model_update_{model_name}"):
                    if mtime <= getattr(self, f"_last_model_update_{model_name}"):
                        continue
                
                # Load the updated model
                try:
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=self._get_custom_objects()
                    )
                    self.models[model_name] = model
                    setattr(self, f"_last_model_update_{model_name}", mtime)
                    logger.info(f"Updated model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to update model {model_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error checking for model updates: {e}")

# Example usage
if __name__ == "__main__":
    # Create a sample configuration
    config = {
        "models": [
            {
                "name": "lstm_micro",
                "model_type": "lstm",
                "input_shape": [60, 8],  # sequence_length x num_features
                "output_shape": 2,       # binary classification
                "timeframes": ["1m", "5m"],
                "indicators": ["rsi", "macd"],
                "sequence_length": 60,
                "use_attention": False,
                "dropout_rate": 0.2,
                "learning_rate": 1e-4
            }
        ]
    }
    
    # Save sample config
    os.makedirs("config", exist_ok=True)
    with open("config/micro_predictor_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize predictor
    predictor = MicroPredictor("config/micro_predictor_config.json")
    
    # Example of adding data (in practice, this would come from market feed)
    sample_data = {
        'open': 100.0,
        'high': 101.0,
        'low': 99.5,
        'close': 100.5,
        'volume': 1000.0
    }
    
    # Add some historical data
    for i in range(1000):
        predictor.add_market_data('1m', sample_data)
    
    # Make a prediction
    try:
        result = predictor.predict(
            prediction_type='direction',
            timeframes=['1m'],
            ensemble_method='average'
        )
        print("\nPrediction Result:")
        print(json.dumps({
            'prediction': result['prediction'].tolist(),
            'confidence': result['confidence'],
            'num_predictions': result['num_predictions']
        }, indent=2))
        
    except Exception as e:
        print(f"Prediction failed: {e}")