import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import joblib
from pathlib import Path
import logging
from datetime import datetime

# Import ML libraries
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Setup logging
logger = logging.getLogger('trading.ai_core')

class AIModel:
    """Base class for AI models in the trading system."""
    
    def __init__(self, model_type: str, params: Optional[Dict] = None):
        """
        Initialize the AI model.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', etc.)
            params: Model hyperparameters
        """
        self.model_type = model_type
        self.params = params or {}
        self.model = self._init_model()
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.trained_at = None
        self.metrics = {}
        
    def _init_model(self):
        """Initialize the specific ML model based on model_type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**self.params)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**self.params)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the input data.
        
        Args:
            X: Input features
            
        Returns:
            Preprocessed features
        """
        return self.scaler.fit_transform(X)
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train the model and evaluate on a test set.
        
        Args:
            X: Training features
            y: Training labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Preprocess
            X_train_scaled = self.preprocess_data(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.trained_at = datetime.utcnow()
            
            # Evaluate
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'test_size': len(X_test),
                'train_size': len(X_train),
                'model_type': self.model_type,
                'trained_at': self.trained_at.isoformat()
            }
            
            # Get feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_.tolist()
                self.metrics['feature_importances'] = self.feature_importances_
            
            logger.info(f"Model training complete. Test accuracy: {self.metrics['accuracy']:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        try:
            model_data = {
                'model_type': self.model_type,
                'params': self.params,
                'metrics': self.metrics,
                'feature_importances': self.feature_importances_,
                'trained_at': self.trained_at.isoformat() if self.trained_at else None,
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'AIModel':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded AIModel instance
        """
        try:
            model_data = joblib.load(filepath)
            instance = cls(
                model_type=model_data['model_type'],
                params=model_data.get('params', {})
            )
            instance.model = model_data['model']
            instance.scaler = model_data['scaler']
            instance.metrics = model_data.get('metrics', {})
            instance.feature_importances_ = model_data.get('feature_importances')
            instance.trained_at = (
                datetime.fromisoformat(model_data['trained_at'])
                if model_data.get('trained_at') else None
            )
            logger.info(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

class EnsembleModel:
    """Ensemble of multiple AI models for improved predictions."""
    
    def __init__(self, models: List[Dict]):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of model configurations
        """
        self.models = []
        self.weights = []
        self.scaler = StandardScaler()
        self.metrics = {}
        
        for model_config in models:
            self.models.append(AIModel(**model_config))
            self.weights.append(model_config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w/total_weight for w in self.weights]
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train all models in the ensemble.
        
        Args:
            X: Training features
            y: Training labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Preprocess
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each model
            all_metrics = {}
            for i, model in enumerate(self.models):
                logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_type}")
                metrics = model.train(X_train_scaled, y_train, test_size=0)
                all_metrics[f"model_{i}"] = metrics
            
            # Evaluate ensemble
            y_pred_proba = np.zeros(len(X_test_scaled))
            for model, weight in zip(self.models, self.weights):
                _, probas = model.predict(X_test_scaled)
                y_pred_proba += probas * weight
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate ensemble metrics
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'test_size': len(X_test),
                'train_size': len(X_train),
                'model_type': 'ensemble',
                'component_metrics': all_metrics,
                'weights': self.weights
            }
            
            logger.info(f"Ensemble training complete. Test accuracy: {self.metrics['accuracy']:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            X_scaled = self.scaler.transform(X)
            y_pred_proba = np.zeros(len(X_scaled))
            
            for model, weight in zip(self.models, self.weights):
                _, probas = model.predict(X_scaled)
                y_pred_proba += probas * weight
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            return y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise
    
    def save(self, filepath: str) -> None:
        """
        Save the ensemble to disk.
        
        Args:
            filepath: Path to save the ensemble
        """
        try:
            ensemble_data = {
                'models': [
                    {
                        'model_type': model.model_type,
                        'params': model.params,
                        'metrics': model.metrics
                    }
                    for model in self.models
                ],
                'weights': self.weights,
                'metrics': self.metrics,
                'scaler': self.scaler
            }
            joblib.dump(ensemble_data, filepath)
            logger.info(f"Ensemble saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleModel':
        """
        Load a saved ensemble from disk.
        
        Args:
            filepath: Path to the saved ensemble
            
        Returns:
            Loaded EnsembleModel instance
        """
        try:
            ensemble_data = joblib.load(filepath)
            instance = cls(ensemble_data['models'])
            instance.weights = ensemble_data['weights']
            instance.metrics = ensemble_data.get('metrics', {})
            instance.scaler = ensemble_data['scaler']
            
            # Load each model
            for i, model_data in enumerate(ensemble_data['models']):
                model_path = filepath.replace('.pkl', f'_model_{i}.pkl')
                if Path(model_path).exists():
                    instance.models[i] = AIModel.load(model_path)
            
            logger.info(f"Ensemble loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            raise

class AICore:
    """Main AI core for the trading system."""
    
    def __init__(self, config: Dict):
        """
        Initialize the AI core.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.current_model = None
        self.model_version = None
        
        # Load or create models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize or load models based on configuration."""
        model_configs = self.config.get('models', [])
        
        for model_config in model_configs:
            model_type = model_config['type']
            model_id = model_config.get('id', model_type)
            
            # Try to load existing model
            model_path = self.models_dir / f"{model_id}.pkl"
            if model_path.exists():
                try:
                    if model_config.get('ensemble', False):
                        self.models[model_id] = EnsembleModel.load(model_path)
                    else:
                        self.models[model_id] = AIModel.load(model_path)
                    logger.info(f"Loaded {model_type} model: {model_id}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_id}: {e}")
                    self._create_new_model(model_config, model_id)
            else:
                self._create_new_model(model_config, model_id)
        
        # Set default model
        default_model = self.config.get('default_model')
        if default_model and default_model in self.models:
            self.current_model = self.models[default_model]
            self.model_version = default_model
            logger.info(f"Default model set to: {default_model}")
    
    def _create_new_model(self, model_config: Dict, model_id: str) -> None:
        """Create a new model instance."""
        try:
            if model_config.get('ensemble', False):
                self.models[model_id] = EnsembleModel(model_config.get('models', []))
            else:
                self.models[model_id] = AIModel(
                    model_type=model_config['type'],
                    params=model_config.get('params', {})
                )
            logger.info(f"Created new {model_config['type']} model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to create model {model_id}: {e}")
            raise
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_id: Optional[str] = None,
        save: bool = True
    ) -> Dict:
        """
        Train a model.
        
        Args:
            X: Training features
            y: Training labels
            model_id: ID of the model to train (uses current model if None)
            save: Whether to save the trained model
            
        Returns:
            Training metrics
        """
        model = self.current_model if model_id is None else self.models.get(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        
        try:
            metrics = model.train(X, y)
            
            if save:
                model_path = self.models_dir / f"{model_id or self.model_version}.pkl"
                model.save(model_path)
                logger.info(f"Model saved to {model_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(
        self,
        X: np.ndarray,
        model_id: Optional[str] = None,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using a model.
        
        Args:
            X: Input features
            model_id: ID of the model to use (uses current model if None)
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        model = self.current_model if model_id is None else self.models.get(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        
        try:
            if return_proba:
                return model.predict(X)
            else:
                preds, _ = model.predict(X)
                return preds
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def set_current_model(self, model_id: str) -> None:
        """
        Set the current model to use for predictions.
        
        Args:
            model_id: ID of the model to set as current
        """
        if model_id in self.models:
            self.current_model = self.models[model_id]
            self.model_version = model_id
            logger.info(f"Current model set to: {model_id}")
        else:
            raise ValueError(f"Model {model_id} not found")
    
    def get_model_metrics(self, model_id: Optional[str] = None) -> Dict:
        """
        Get metrics for a model.
        
        Args:
            model_id: ID of the model (uses current model if None)
            
        Returns:
            Dictionary of model metrics
        """
        model = self.current_model if model_id is None else self.models.get(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        
        return model.metrics if hasattr(model, 'metrics') else {}

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'models_dir': 'models',
        'default_model': 'ensemble_v1',
        'models': [
            {
                'id': 'random_forest',
                'type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'random_state': 42
                }
            },
            {
                'id': 'gradient_boosting',
                'type': 'gradient_boosting',
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            {
                'id': 'ensemble_v1',
                'ensemble': True,
                'models': [
                    {
                        'type': 'random_forest',
                        'params': {'n_estimators': 100, 'max_depth': 5},
                        'weight': 0.5
                    },
                    {
                        'type': 'gradient_boosting',
                        'params': {'n_estimators': 100, 'learning_rate': 0.1},
                        'weight': 0.5
                    }
                ]
            }
        ]
    }
    
    # Initialize AI core
    ai_core = AICore(config)
    
    # Example training data (replace with your actual data)
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification
    
    # Train the ensemble model
    metrics = ai_core.train_model(X, y, model_id='ensemble_v1')
    print(f"Training metrics: {metrics}")
    
    # Make predictions
    X_test = np.random.rand(10, 10)  # 10 test samples
    predictions = ai_core.predict(X_test)
    print(f"Predictions: {predictions}")