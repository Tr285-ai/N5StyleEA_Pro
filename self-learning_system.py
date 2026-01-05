# learning_system.py
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger("LearningSystem")

class SelfLearningSystem:
    def __init__(self, model_registry, config: dict):
        self.model_registry = model_registry
        self.config = config
        self.trade_history = []
        self.model_performance = {}
        self.scaler = StandardScaler()
        self._load_state()
        
    def _load_state(self):
        """Load learning system state from disk"""
        try:
            if os.path.exists('learning_state.joblib'):
                state = joblib.load('learning_state.joblib')
                self.trade_history = state.get('trade_history', [])
                self.model_performance = state.get('model_performance', {})
                if 'scaler' in state:
                    self.scaler = state['scaler']
        except Exception as e:
            logger.warning(f"Could not load learning state: {e}")

    def _save_state(self):
        """Save learning system state to disk"""
        try:
            joblib.dump({
                'trade_history': self.trade_history,
                'model_performance': self.model_performance,
                'scaler': self.scaler
            }, 'learning_state.joblib')
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    def record_trade_outcome(self, trade_result: dict):
        """Record trade outcome and update models"""
        try:
            self.trade_history.append({
                **trade_result,
                'recorded_at': datetime.utcnow().isoformat()
            })
            
            # Update model performance
            model_id = trade_result.get('model_id')
            if model_id:
                if model_id not in self.model_performance:
                    self.model_performance[model_id] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0.0,
                        'recent_performance': []
                    }
                
                perf = self.model_performance[model_id]
                perf['total_trades'] += 1
                if trade_result.get('profit', 0) > 0:
                    perf['winning_trades'] += 1
                perf['total_pnl'] += trade_result.get('profit', 0)
                
                # Keep recent performance (last 100 trades)
                perf['recent_performance'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'profit': trade_result.get('profit', 0),
                    'confidence': trade_result.get('confidence', 0)
                })
                perf['recent_performance'] = perf['recent_performance'][-100:]
                
                # Update model weights if needed
                if len(self.trade_history) % self.config.get('retrain_interval', 50) == 0:
                    self._update_model_weights()
            
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")

    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        try:
            if not self.trade_history:
                return
                
            # Convert trade history to DataFrame
            df = pd.DataFrame(self.trade_history)
            if len(df) < 10:  # Need enough data
                return
                
            # Prepare features and target
            X, y = self._prepare_training_data(df)
            if X is None or len(X) < 10:
                return
                
            # Train a meta-model to predict trade outcomes
            meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
            meta_model.fit(X, y)
            
            # Save the updated model
            model_type = "meta_learner"
            metrics = {
                'r2_score': meta_model.score(X, y),
                'trained_at': datetime.utcnow().isoformat(),
                'num_samples': len(X)
            }
            
            self.model_registry.register_model(
                model_type=model_type,
                model=meta_model,
                metrics=metrics
            )
            
            logger.info(f"Updated meta-model with {len(X)} training samples")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for meta-learning"""
        try:
            # Add time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            
            # Calculate features
            features = ['hour', 'day_of_week', 'confidence', 'risk_reward_ratio']
            features = [f for f in features if f in df.columns]
            
            X = df[features].values
            y = df['profit'].values
            
            # Scale features
            if len(X) > 0:
                if hasattr(self.scaler, 'n_features_in_'):
                    X = self.scaler.transform(X)
                else:
                    self.scaler.fit(X)
                    X = self.scaler.transform(X)
                    
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None

    def get_trade_suggestions(self, market_data: dict, model_id: str) -> dict:
        """Get trade suggestions based on current market data"""
        try:
            # Load the latest model
            model_info = self.model_registry.get_latest_model("trade_predictor")
            if not model_info:
                return {"error": "No trained model available"}
                
            model = joblib.load(model_info['path'])
            
            # Prepare input features
            features = self._prepare_prediction_features(market_data)
            if features is None:
                return {"error": "Could not prepare features"}
                
            # Make prediction
            prediction = model.predict([features])[0]
            
            return {
                "action": "BUY" if prediction > 0.5 else "SELL",
                "confidence": float(prediction),
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting trade suggestions: {e}")
            return {"error": str(e)}

    def _prepare_prediction_features(self, market_data: dict) -> Optional[np.ndarray]:
        """Prepare features for prediction"""
        try:
            # Extract relevant features from market data
            features = [
                market_data.get('hour', 0),
                market_data.get('day_of_week', 0),
                market_data.get('volatility', 0),
                market_data.get('volume', 0),
                market_data.get('rsi', 50),
                market_data.get('atr', 0)
            ]
            
            # Scale features
            if hasattr(self.scaler, 'n_features_in_'):
                return self.scaler.transform([features])[0]
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return None