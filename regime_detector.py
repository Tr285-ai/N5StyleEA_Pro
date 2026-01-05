# regime_detector.py
"""
Regime Detector v15.2 - Enhanced

Provides intelligent market regime detection with the following features:
- Multiple regime classification (trending, ranging, volatile)
- Volatility clustering detection
- Momentum shift analysis
- Liquidity pocket identification
- Market shift detection
- Model training and persistence

Author: N5StyleEA Team
Version: 15.2.1
"""

import os
import math
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional ML imports with graceful fallback
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("XGBoost not available. Some features may be limited.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline
    _HAS_SK = True
except ImportError:
    _HAS_SK = False
    logger.warning("scikit-learn not available. ML features will be disabled.")

# Constants
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATHS = {
    'model': os.path.join(MODELS_DIR, "regime_model_v15.2.pkl"),
    'scaler': os.path.join(MODELS_DIR, "regime_scaler_v15.2.pkl"),
    'metadata': os.path.join(MODELS_DIR, "regime_metadata_v15.2.json")
}

# Default parameters
DEFAULT_LOOKBACK = 60
DEFAULT_VOLATILITY_THRESHOLD = 0.7
DEFAULT_MOMENTUM_THRESHOLD = 0.0008
DEFAULT_VOLUME_SPIKE_THRESHOLD = 3.0

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    VOLATILE = auto()
    UNKNOWN = auto()

class LiquidityLevel(Enum):
    """Liquidity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class RegimePrediction:
    """Container for regime prediction results."""
    regime: MarketRegime
    confidence: float
    features: Dict[str, float]
    metadata: Dict[str, Any] = None

@dataclass
class MarketShift:
    """Container for market shift detection results."""
    shift_detected: bool
    reason: str
    skip_seconds: int
    details: Dict[str, Any]

class RegimeDetector:
    """
    Advanced market regime detection system.
    
    Features:
    - Multiple regime classification (trending, ranging, volatile)
    - Volatility clustering detection
    - Momentum shift analysis
    - Liquidity pocket identification
    - Market shift detection
    """
    
    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        metadata_path: str = None,
        lookback: int = DEFAULT_LOOKBACK
    ):
        """
        Initialize the regime detector.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            metadata_path: Path to model metadata
            lookback: Number of candles to use for analysis
        """
        self.model = None
        self.scaler = None
        self.metadata = {}
        self.lookback = lookback
        self._last_prediction = None
        
        # Load model if paths not provided
        model_path = model_path or MODEL_PATHS['model']
        scaler_path = scaler_path or MODEL_PATHS['scaler']
        metadata_path = metadata_path or MODEL_PATHS['metadata']
        
        self._load_model(model_path, scaler_path, metadata_path)
    
    def _load_model(
        self,
        model_path: str,
        scaler_path: str,
        metadata_path: str
    ) -> None:
        """Load pre-trained model, scaler, and metadata."""
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Loaded model from {model_path}")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
                
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.metadata = {}
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features for regime detection.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of feature values
        """
        if df is None or len(df) < 10:
            return {}
            
        try:
            d = df.tail(self.lookback).copy()
            if len(d) < 10:  # Minimum required candles
                return {}
                
            closes = d["close"].astype(float).values
            highs = d["high"].astype(float).values
            lows = d["low"].astype(float).values
            volumes = d["volume"].astype(float).values if "volume" in d.columns else np.ones(len(d))
            
            # Price returns
            returns = np.diff(closes) / (closes[:-1] + 1e-9)
            
            # Basic statistics
            features = {
                'returns_mean': float(np.mean(returns)) if len(returns) > 0 else 0.0,
                'returns_std': float(np.std(returns)) if len(returns) > 0 else 0.0,
                'returns_skew': float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0,
                'returns_kurt': float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0.0,
            }
            
            # Momentum
            n = min(20, len(closes))
            if n >= 2:
                x = np.arange(n)
                y = closes[-n:]
                slope = np.polyfit(x, y, 1)[0]
                features['momentum'] = float(slope / (np.mean(y) + 1e-9))
            else:
                features['momentum'] = 0.0
            
            # Volatility (ATR)
            trs = []
            for i in range(1, len(d)):
                prev_close = closes[i-1]
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - prev_close)
                low_close = abs(lows[i] - prev_close)
                trs.append(max(high_low, high_close, low_close))
            features['atr'] = float(np.mean(trs)) if trs else 0.0
            
            # Volume analysis
            vol_median = float(np.median(volumes[:-1])) if len(volumes) > 1 else 1.0
            features['volume_ratio'] = float(volumes[-1] / (vol_median + 1e-9))
            
            # Additional features
            features['range_ratio'] = float((highs[-1] - lows[-1]) / (np.mean(highs - lows) + 1e-9))
            features['close_to_range'] = float((closes[-1] - lows[-1]) / ((highs[-1] - lows[-1]) + 1e-9))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def detect_regime(self, df: pd.DataFrame) -> RegimePrediction:
        """
        Detect the current market regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            RegimePrediction object
        """
        features = self.extract_features(df)
        if not features:
            return RegimePrediction(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={}
            )
            
        try:
            # Prepare features for model
            feature_names = sorted(features.keys())
            X = np.array([features[k] for k in feature_names]).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
                
            # Make prediction if model is available
            if self.model is not None:
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X_scaled)[0]
                    pred_class = np.argmax(proba)
                    confidence = float(proba[pred_class])
                else:
                    pred_class = self.model.predict(X_scaled)[0]
                    confidence = 1.0  # Default confidence for non-probabilistic models
            else:
                # Fallback to rule-based detection
                return self._rule_based_regime(features)
                
            # Map prediction to MarketRegime enum
            regime_map = {
                0: MarketRegime.TRENDING_UP,
                1: MarketRegime.TRENDING_DOWN,
                2: MarketRegime.RANGING,
                3: MarketRegime.VOLATILE
            }
            regime = regime_map.get(pred_class, MarketRegime.UNKNOWN)
            
            # Store last prediction
            self._last_prediction = RegimePrediction(
                regime=regime,
                confidence=confidence,
                features=features,
                metadata={
                    'model': self.model.__class__.__name__ if self.model else 'rule_based',
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            return self._last_prediction
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return RegimePrediction(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features=features,
                metadata={'error': str(e)}
            )
    
    def _rule_based_regime(self, features: Dict[str, float]) -> RegimePrediction:
        """Fallback rule-based regime detection."""
        momentum = features.get('momentum', 0)
        atr = features.get('atr', 0)
        returns_std = features.get('returns_std', 0)
        
        # Simple rule-based classification
        if abs(momentum) > 0.001 and atr > 0.0005:
            regime = MarketRegime.TRENDING_UP if momentum > 0 else MarketRegime.TRENDING_DOWN
            confidence = min(0.8, abs(moment