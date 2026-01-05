# expiry_selector.py
"""
Intelligent Expiry Selector for N5StyleEA Pro v15.2

This module provides intelligent expiry time selection for binary options trading.
It combines rule-based and machine learning approaches to determine optimal
expiry times based on market conditions.

Features:
- Rule-based fallback for cold start
- Machine learning model for improved accuracy
- Volatility and momentum-based adjustments
- Confidence scoring for expiry selection
- Model training and persistence

Author: N5StyleEA Team
Version: 15.2
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "models"
EXPIRY_KMEANS_PATH = os.path.join(MODEL_DIR, "expiry_kmeans_v15.2.pkl")
EXPIRY_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "expiry_classifier_v15.2.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# Default expiry buckets in seconds
DEFAULT_EXPIRY_BUCKETS = [60, 120, 180, 240, 300]  # 1m, 2m, 3m, 4m, 5m

# Rule thresholds (tunable)
DEFAULT_THRESHOLDS = {
    'ATR_FAST': 0.0008,    # Fast market threshold (1m ATR)
    'ATR_MEDIUM': 0.0004,  # Medium market threshold
    'VOLATILITY_HIGH': 0.0015,
    'VOLATILITY_MEDIUM': 0.0008,
    'MOMENTUM_STRONG': 0.0007,
    'MIN_CONFIDENCE': 0.55,
    'ML_OVERRIDE_THRESHOLD': 0.65
}

@dataclass
class ExpirySelection:
    """Container for expiry selection results."""
    expiry_seconds: int
    confidence: float
    method: str
    reason: str
    features: Optional[Dict[str, float]] = None
    ml_confidence: Optional[float] = None
    rule_confidence: Optional[float] = None

class ExpirySelector:
    """
    Intelligent expiry time selector for binary options trading.
    Combines rule-based and machine learning approaches.
    """
    
    def __init__(
        self,
        expiry_buckets: List[int] = None,
        thresholds: Dict[str, float] = None,
        use_ml: bool = True
    ):
        """
        Initialize the expiry selector.
        
        Args:
            expiry_buckets: List of possible expiry times in seconds
            thresholds: Dictionary of threshold values for rule-based selection
            use_ml: Whether to use machine learning for predictions
        """
        self.expiry_buckets = expiry_buckets or DEFAULT_EXPIRY_BUCKETS
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.use_ml = use_ml
        self.kmeans_model = None
        self.classifier = None
        
        # Load ML models if available
        if self.use_ml:
            self._load_models()
    
    def _load_models(self) -> None:
        """Load pre-trained ML models if they exist."""
        try:
            if os.path.exists(EXPIRY_KMEANS_PATH):
                self.kmeans_model = joblib.load(EXPIRY_KMEANS_PATH)
                logger.info("Loaded KMeans model for expiry selection")
                
            if os.path.exists(EXPIRY_CLASSIFIER_PATH):
                self.classifier = joblib.load(EXPIRY_CLASSIFIER_PATH)
                logger.info("Loaded classifier model for expiry selection")
                
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self.kmeans_model = None
            self.classifier = None
    
    def _compute_features(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """
        Compute technical features from candle data.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of candles to use for feature calculation
            
        Returns:
            Dictionary of feature values
        """
        if df.empty or len(df) < 3:
            return {
                'atr': 0.0,
                'vol_z': 0.0,
                'range_ratio': 0.0,
                'momentum': 0.0,
                'tick_speed': 1.0,
                'volatility': 0.0
            }
            
        df = df.copy().tail(lookback).reset_index(drop=True)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change().fillna(0)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Volume Z-Score
        vol_mean = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else 0
        vol_std = df['volume'].rolling(20).std().iloc[-1] if 'volume' in df.columns else 1
        vol_z = (df['volume'].iloc[-1] - vol_mean) / (vol_std if vol_std > 0 else 1)
        
        # Range ratio (current range vs average range)
        avg_range = (df['high'] - df['low']).rolling(20).mean().iloc[-1]
        current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        range_ratio = current_range / (avg_range if avg_range > 0 else 1e-9)
        
        # Momentum (price change over last 3 candles)
        mom = (df['close'].iloc[-1] - df['close'].iloc[-3]) / (df['close'].iloc[-3] + 1e-9)
        
        # Volatility (standard deviation of returns)
        volatility = df['returns'].std() * math.sqrt(252 * 24 * 60)  # Annualized
        
        return {
            'atr': float(atr),
            'vol_z': float(vol_z),
            'range_ratio': float(range_ratio),
            'momentum': float(mom),
            'tick_speed': 1.0,  # Placeholder for actual tick data
            'volatility': float(volatility)
        }
    
    def _rule_based_selection(self, features: Dict[str, float]) -> Tuple[int, float, str]:
        """
        Rule-based expiry selection.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (expiry_seconds, confidence, reason)
        """
        atr = features.get('atr', 0)
        vol_z = features.get('vol_z', 0)
        range_ratio = features.get('range_ratio', 0)
        momentum = abs(features.get('momentum', 0))
        
        # Default values
        choice = 180  # Default to 3 minutes
        confidence = self.thresholds['MIN_CONFIDENCE']
        reason = []
        
        # High volatility or volume spike -> shorter expiry
        if (atr >= self.thresholds['ATR_FAST'] or 
            vol_z >= 2.0 or 
            range_ratio >= 1.5):
            
            if momentum > self.thresholds['MOMENTUM_STRONG']:
                choice = min(self.expiry_buckets)  # Shortest expiry
                reason.append("strong_momentum_high_volatility")
            else:
                choice = min(x for x in self.expiry_buckets if x >= 60)  # At least 1 minute
                reason.append("high_volatility")
            confidence = 0.7
            
        # Medium volatility
        elif atr >= self.thresholds['ATR_MEDIUM']:
            choice = 180  # 3 minutes
            reason.append("medium_volatility")
            confidence = 0.6
            
        # Low volatility -> longer expiry
        else:
            choice = max(self.expiry_buckets)  # Longest expiry
            reason.append("low_volatility")
            confidence = 0.65
            
        # Adjust confidence based on feature consistency
        if (vol_z > 1.5 and atr > self.thresholds['ATR_MEDIUM'] and 
            range_ratio > 1.2 and momentum > 0.0005):
            confidence = min(0.9, confidence + 0.15)
            reason.append("high_confidence_features")
            
        return choice, confidence, "_".join(reason)
    
    def _ml_based_selection(self, features: Dict[str, float]) -> Optional[Tuple[int, float]]:
        """
        Machine learning based expiry selection.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (expiry_seconds, confidence) or None if ML is not available
        """
        if not self.use_ml or (self.kmeans_model is None and self.classifier is None):
            return None
            
        try:
            # Prepare feature vector
            feature_vec = np.array([
                features['atr'],
                features['vol_z'],
                features['range_ratio'],
                features['momentum'],
                features['tick_speed']
            ]).reshape(1, -1)
            
            # Use classifier if available
            if self.classifier is not None:
                if hasattr(self.classifier, 'predict_proba'):
                    probs = self.classifier.predict_proba(feature_vec)[0]
                    bucket_idx = np.argmax(probs)
                    confidence = float(probs[bucket_idx])
                else:
                    bucket_idx = self.classifier.predict(feature_vec)[0]
                    confidence = 0.7  # Default confidence for non-probabilistic classifiers
                
                if 0 <= bucket_idx < len(self.expiry_buckets):
                    return self.expiry_buckets[bucket_idx], confidence
                    
            # Fall back to KMeans if classifier not available
            elif self.kmeans_model is not None:
                cluster = self.kmeans_model.predict(feature_vec)[0]
                centers = self.kmeans_model.cluster_centers_
                
                # Map cluster to expiry (higher volatility -> shorter expiry)
                atrs = centers[:, 0]  # ATR is first feature
                sorted_indices = np.argsort(atrs)
                
                # Map clusters to expiries (volatile clusters -> short expiries)
                cluster_rank = np.where(sorted_indices == cluster)[0][0]
                bucket_idx = min(len(self.expiry_buckets) - 1, 
                               len(self.expiry_buckets) * cluster_rank // len(centers))
                
                # Calculate confidence based on distance to centroid
                centroid = centers[cluster]
                distance = np.linalg.norm(feature_vec - centroid)
                confidence = max(0.2, min(0.95, 1.0 / (1.0 + distance)))
                
                return self.expiry_buckets[bucket_idx], confidence
                
        except Exception as e:
            logger.error(f"Error in ML-based selection: {e}")
            
        return None
    
    def select_expiry(
        self,
        df_candles: pd.DataFrame,
        symbol: Optional[str] = None,
        lookback: int = 20
    ) -> ExpirySelection:
        """
        Select the optimal expiry time based on current market conditions.
        
        Args:
            df_candles: DataFrame with OHLCV data
            symbol: Optional symbol for context
            lookback: Number of candles to use for feature calculation
            
        Returns:
            ExpirySelection object with results
        """
        # Compute features
        features = self._compute_features(df_candles, lookback)
        
        # Get rule-based selection
        rule_choice, rule_confidence, rule_reason = self._rule_based_selection(features)
        
        # Initialize result with rule-based selection
        result = ExpirySelection(
            expiry_seconds=rule_choice,
            confidence=rule_confidence,
            method="rule_based",
            reason=rule_reason,
            features=features,
            rule_confidence=rule_confidence
        )
        
        # Try ML-based selection
        if self.use_ml:
            ml_result = self._ml_based_selection(features)
            if ml_result:
                ml_choice, ml_confidence = ml_result
                
                # Only override if ML confidence is high enough
                if ml_confidence >= self.thresholds['ML_OVERRIDE_THRESHOLD']:
                    result.expiry_seconds = ml_choice
                    result.confidence = ml_confidence
                    result.method = "ml_override"
                    result.ml_confidence = ml_confidence
                    result.reason = f"ml_override_{result.reason}"
        
        # Ensure expiry is within bounds
        result.expiry_seconds = max(min(result.expiry_seconds, max(self.expiry_buckets)), 
                                   min(self.expiry_buckets))
        
        return result

# Helper functions for backward compatibility
def choose_expiry(
    df_candles: pd.DataFrame, 
    symbol: Optional[str] = None, 
    lookback: int = 20
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        df_candles: DataFrame with OHLCV data
        symbol: Optional symbol for context
        lookback: Number of candles to use for feature calculation
        
    Returns:
        Tuple of (expiry_seconds, confidence, result_dict)
    """
    selector = ExpirySelector()
    result = selector.select_expiry(df_candles, symbol, lookback)
    
    # Convert to legacy format
    result_dict = {
        'method': result.method,
        'reason': result.reason,
        'features': result.features,
        'ml_confidence': result.ml_confidence
    }
    
    return result.expiry_seconds, result.confidence, result_dict

def train_expiry_model(
    data_folder: str = "data",
    n_clusters: int = 3,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Train and save the expiry selection models.
    
    Args:
        data_folder: Directory containing training data
        n_clusters: Number of clusters for KMeans
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report
        
        # Load and prepare data
        signals_path = os.path.join(data_folder, "signals_log.csv")
        if not os.path.exists(signals_path):
            raise FileNotFoundError(f"signals_log.csv not found in {data_folder}")
            
        signals = pd.read_csv(signals_path)
        required_cols = ['timestamp', 'symbol', 'expiry', 'outcome']
        if not all(col in signals.columns for col in required_cols):
            raise ValueError(f"signals_log.csv must contain columns: {required_cols}")
        
        # Feature engineering
        X = []
        y = []
        
        for _, row in signals.iterrows():
            symbol = row['symbol']
            candle_file = os.path.join(data_folder, f"{symbol}_candles.csv")
            
            if not os.path.exists(candle_file):
                continue
                
            try:
                # Load candle data
                candles = pd.read_csv(candle_file)
                if len(candles) < 30:  # Minimum candles needed
                    continue
                    
                # Compute features
                selector = ExpirySelector()
                features = selector._compute_features(candles)
                
                # Skip if any feature is missing
                if any(math.isnan(v) or math.isinf(v) for v in features.values()):
                    continue
                    
                X.append(list(features.values()))
                
                # Convert expiry to bucket index
                expiry = int(row['expiry'])
                closest = min(selector.expiry_buckets, key=lambda x: abs(x - expiry))
                y.append(selector.expiry_buckets.index(closest))
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not X or not y:
            raise ValueError("No valid training data found")
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(X_train_scaled)
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = clf.score(X_train_scaled, y_train)
        test_score = clf.score(X_test_scaled, y_test)
        
        logger.info(f"Train accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Save models
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(kmeans, EXPIRY_KMEANS_PATH)
        joblib.dump(clf, EXPIRY_CLASSIFIER_PATH)
        
        logger.info(f"Models saved to {MODEL_DIR}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(
                list(features.keys()),
                clf.feature_importances_.tolist()
            ))
        }
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    df = pd.DataFrame({
        'open': np.random.normal(1.1000, 0.001, 100).cumsum(),
        'high': np.random.normal(1.1010, 0.001, 100).cumsum(),
        'low': np.random.normal(1.0990, 0.001, 100).cumsum(),
        'close': np.random.normal(1.1005, 0.001, 100).cumsum(),
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    # Initialize selector
    selector = ExpirySelector()
    
    # Select expiry
    result = selector.select_expiry(df)
    
    print(f"Selected expiry: {result.expiry_seconds} seconds")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Method: {result.method}")
    print(f"Reason: {result.reason}")
    print(f"Features: {result.features}")