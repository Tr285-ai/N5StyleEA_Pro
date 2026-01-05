# ensemble_model.py
"""
Ensemble Model for N5StyleEA v15.2

Provides ensemble learning capabilities combining multiple models for improved predictions.
"""
import os
import math
import joblib
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import xgboost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    _HAS_XGB = True
    logger.info("XGBoost successfully imported")
except ImportError as e:
    _HAS_XGB = False
    logger.warning(f"XGBoost not available, using RandomForest only: {e}")

# Constants
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "ensemble_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "ensemble_scaler.pkl")
META_PATH = os.path.join(MODELS_DIR, "ensemble_meta.json")
EPSILON = 1e-9

# [Previous functions like features_from_df would go here]
# [Previous implementation remains exactly the same]

class EnsembleTradingModel:
    """
    Ensemble model for trading signal prediction.
    Combines multiple base models to make more robust predictions.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, 
                 scaler_path: str = SCALER_PATH,
                 meta_path: str = META_PATH):
        """
        Initialize the ensemble model.
        
        Args:
            model_path: Path to save/load the model
            scaler_path: Path to save/load the feature scaler
            meta_path: Path to save/load model metadata
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.meta_path = meta_path
        self.model = None
        self.scaler = StandardScaler()
        self.metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "model_type": "Ensemble",
            "features": None,
            "performance": {}
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, 
             test_size: float = 0.2,
             random_state: int = 42) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results
        """
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=test_size,
                random_state=random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Initialize base models
            models = [
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=random_state,
                    n_jobs=-1
                ))
            ]
            
            # Add XGBoost if available
            if _HAS_XGB:
                models.append(('xgb', XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=random_state
                )))
            
            # Create ensemble
            self.model = VotingClassifier(
                estimators=models,
                voting='soft',
                n_jobs=-1
            )
            
            # Train
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, self.model.predict(X_train))
            test_acc = accuracy_score(y_test, self.model.predict(X_test))
            
            # Calculate ROC AUC if we have more than one class
            if len(np.unique(y)) > 1:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
                
            # Update metadata
            self.metadata.update({
                "trained_at": datetime.utcnow().isoformat(),
                "performance": {
                    "train_accuracy": float(train_acc),
                    "test_accuracy": float(test_acc),
                    "roc_auc": float(roc_auc) if roc_auc is not None else None
                },
                "data_stats": {
                    "n_samples": len(X),
                    "n_features": X.shape[1],
                    "class_distribution": dict(zip(*np.unique(y, return_counts=True)))
                }
            })
            
            return self.metadata["performance"]
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def save(self) -> Dict[str, str]:
        """Save the model and metadata."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            # Save scaler
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save metadata
            with open(self.meta_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Model saved to {self.model_path}")
            return {
                "model": self.model_path,
                "scaler": self.scaler_path,
                "metadata": self.meta_path
            }
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)
            raise
            
    @classmethod
    def load(cls, model_path: str = MODEL_PATH,
                    scaler_path: str = SCALER_PATH,
                    meta_path: str = META_PATH) -> 'EnsembleTradingModel':
        """Load a trained model from disk."""
        try:
            if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
                raise FileNotFoundError(
                    f"Model files not found. Check paths: {model_path}, {scaler_path}, {meta_path}"
                )
                
            # Create instance
            ensemble = cls(model_path, scaler_path, meta_path)
            
            # Load components
            ensemble.model = joblib.load(model_path)
            ensemble.scaler = joblib.load(scaler_path)
            
            # Load metadata
            with open(meta_path, 'r') as f:
                ensemble.metadata = json.load(f)
                
            logger.info(f"Model loaded from {model_path}")
            return ensemble
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Example usage
        logger.info("Ensemble model example")
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = (X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3 + np.random.randn(1000) * 0.1) > 0
        
        # Initialize and train
        model = EnsembleTradingModel()
        logger.info("Training model...")
        metrics = model.train(X, y)
        
        # Show metrics
        logger.info(f"Training complete. Metrics: {metrics}")
        
        # Save model
        model.save()
        
        # Load model
        loaded_model = EnsembleTradingModel.load()
        logger.info("Model loaded successfully")
        
        # Make predictions
        sample = X[:5]
        predictions = loaded_model.predict(sample)
        probabilities = loaded_model.predict_proba(sample)
        
        logger.info(f"Sample predictions: {predictions}")
        logger.info(f"Sample probabilities: {probabilities}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)