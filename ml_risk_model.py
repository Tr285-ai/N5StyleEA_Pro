# risk/ml_risk_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import joblib
import os

class MLRiskModel:
    def __init__(self, model_path: str = None):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.model_path = model_path or "models/risk_model.joblib"
        self.is_fitted = False
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            self.load_model()

    def preprocess_data(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare market data for the model."""
        features = market_data.copy()
        
        # Calculate returns and volatility
        features['returns'] = features['close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        features.dropna(inplace=True)
        
        # Select and scale features
        feature_columns = ['returns', 'volatility', 'volume']
        X = features[feature_columns]
        
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled

    def train(self, market_data: pd.DataFrame):
        """Train the risk model on historical data."""
        X = self.preprocess_data(market_data)
        self.model.fit(X)
        self.is_fitted = True
        self.save_model()
        
    def predict_risk(self, market_data: pd.DataFrame) -> Tuple[float, Dict]:
        """Predict risk score and get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
            
        X = self.preprocess_data(market_data)
        risk_scores = self.model.decision_function(X)
        avg_risk = float(np.mean(risk_scores))
        
        # Get feature importance
        importances = self.model.estimators_[0].feature_importances_
        feature_importance = dict(zip(
            ['returns', 'volatility', 'volume'],
            importances.tolist()
        ))
        
        return avg_risk, feature_importance
        
    def save_model(self):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': True
        }, self.model_path)
        
    def load_model(self):
        """Load a trained model from disk."""
        saved = joblib.load(self.model_path)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.is_fitted = saved['is_fitted']