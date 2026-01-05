"""
Expiry Time Selector v15.2

Advanced expiry time selection module for binary options and CFD trading.
Implements both rule-based and ML-based approaches for optimal expiry time selection.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import talib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('expiry_selector.log')
    ]
)
logger = logging.getLogger('expiry_selector')

# Type aliases
Array = np.ndarray
DataFrame = pd.DataFrame

class ExpiryType(Enum):
    """Supported expiry time types."""
    MINUTES = 'minutes'
    HOURS = 'hours'
    DAYS = 'days'
    WEEKS = 'weeks'
    MONTHS = 'months'
    END_OF_DAY = 'end_of_day'
    END_OF_WEEK = 'end_of_week'
    END_OF_MONTH = 'end_of_month'

@dataclass
class ExpiryConfig:
    """Configuration for expiry time selection."""
    min_expiry: int = 1
    max_expiry: int = 60  # in minutes
    default_expiry: int = 5  # in minutes
    expiry_steps: List[int] = field(default_factory=lambda: [1, 2, 5, 15, 30, 60])
    volatility_lookback: int = 14
    volume_lookback: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    use_ml: bool = True
    model_path: str = 'models/expiry_model.joblib'
    retrain_interval: int = 7  # days
    last_retrain_date: Optional[datetime] = None

@dataclass
class MarketConditions:
    """Current market conditions."""
    volatility: float
    trend_strength: float
    volume_ratio: float
    atr: float
    rsi: float
    time_to_news: Optional[float] = None
    current_time: Optional[datetime] = None

class AdvancedExpirySelector:
    """
    Advanced Expiry Time Selector for trading strategies.
    
    This class provides methods to determine optimal expiry times for trades
    based on current market conditions, using both rule-based and ML approaches.
    """
    
    def __init__(self, config: Optional[ExpiryConfig] = None):
        """
        Initialize the expiry selector.
        
        Args:
            config: Configuration for expiry selection. If None, uses defaults.
        """
        self.config = config if config is not None else ExpiryConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'volatility', 'trend_strength', 'volume_ratio', 
            'atr', 'rsi', 'time_to_news'
        ]
        self._initialize_model()
        logger.info("Initialized AdvancedExpirySelector")
    
    def _initialize_model(self) -> None:
        """Initialize or load the ML model."""
        try:
            if os.path.exists(self.config.model_path):
                self._load_model()
                logger.info(f"Loaded model from {self.config.model_path}")
            else:
                self._create_new_model()
                logger.info("Created new model")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self.config.use_ml = False
            logger.warning("Falling back to rule-based selection")
    
    def _create_new_model(self) -> None:
        """Create a new ML model with default parameters."""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            ))
        ])
    
    def _load_model(self) -> None:
        """Load a pre-trained model from disk."""
        try:
            self.model = joblib.load(self.config.model_path)
            logger.info(f"Successfully loaded model from {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the current model to disk."""
        if path is None:
            path = self.config.model_path
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def calculate_market_conditions(
        self,
        df: DataFrame,
        current_time: Optional[datetime] = None
    ) -> MarketConditions:
        """
        Calculate current market conditions from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            current_time: Current time (for news/event timing)
            
        Returns:
            MarketConditions object with current market metrics
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        # Calculate volatility (standard deviation of returns)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate trend strength using ADX
        high, low, close = df['high'], df['low'], df['close']
        adx = talib.ADX(high, low, close, timeperiod=14)
        trend_strength = adx.iloc[-1] / 100.0  # Normalize to [0, 1]
        
        # Calculate volume ratio (current volume vs average)
        volume = df['volume']
        avg_volume = volume.rolling(window=self.config.volume_lookback).mean()
        volume_ratio = (volume.iloc[-1] / avg_volume.iloc[-1]) if avg_volume.iloc[-1] > 0 else 1.0
        
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=self.config.atr_period).iloc[-1]
        
        # Calculate RSI
        rsi = talib.RSI(close, timeperiod=self.config.rsi_period).iloc[-1] / 100.0  # Normalize to [0, 1]
        
        # Time to next news/event (placeholder - implement actual news integration)
        time_to_news = None  # Implement news integration
        
        return MarketConditions(
            volatility=float(volatility),
            trend_strength=float(trend_strength),
            volume_ratio=float(volume_ratio),
            atr=float(atr),
            rsi=float(rsi),
            time_to_news=time_to_news,
            current_time=current_time or datetime.now()
        )
    
    def get_optimal_expiry_rule_based(
        self,
        market: MarketConditions,
        available_expiries: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Determine optimal expiry using rule-based approach.
        
        Args:
            market: Current market conditions
            available_expiries: List of available expiry times in minutes
            
        Returns:
            Tuple of (optimal_expiry, confidence_score)
        """
        if available_expiries is None:
            available_expiries = self.config.expiry_steps
        
        # Base score (0-1) where higher is better for longer expiries
        base_scores = {exp: min(exp / max(available_expiries), 1.0) for exp in available_expiries}
        
        # Adjust based on volatility (higher volatility -> shorter expiry)
        vol_adjustment = 1.0 - min(market.volatility * 5, 0.8)  # Cap at 0.8 reduction
        
        # Adjust based on trend strength (stronger trend -> longer expiry)
        trend_adjustment = 0.5 + (market.trend_strength * 0.5)
        
        # Adjust based on volume (higher volume -> can use shorter expiry)
        volume_adjustment = 1.2 - min(market.volume_ratio, 1.5) / 1.5
        
        # Calculate final scores
        final_scores = {}
        for exp, score in base_scores.items():
            # Apply adjustments
            adj_score = score * vol_adjustment * trend_adjustment * volume_adjustment
            
            # Ensure we're within min/max bounds
            if exp < self.config.min_expiry or exp > self.config.max_expiry:
                adj_score = 0
            
            final_scores[exp] = adj_score
        
        # Get best expiry
        if not final_scores:
            return self.config.default_expiry, 0.5
        
        best_expiry = max(final_scores.items(), key=lambda x: x[1])
        max_score = max(final_scores.values())
        
        # Normalize confidence to [0.5, 1.0] range
        confidence = 0.5 + (0.5 * (max_score / max(1.0, max(final_scores.values()))))
        
        return best_expiry[0], min(max(confidence, 0.5), 1.0)
    
    def get_optimal_expiry_ml(
        self,
        market: MarketConditions,
        available_expiries: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Determine optimal expiry using ML model.
        
        Args:
            market: Current market conditions
            available_expiries: List of available expiry times in minutes
            
        Returns:
            Tuple of (optimal_expiry, confidence_score)
        """
        if not self.config.use_ml or self.model is None:
            return self.get_optimal_expiry_rule_based(market, available_expiries)
        
        try:
            # Prepare features
            features = np.array([
                market.volatility,
                market.trend_strength,
                market.volume_ratio,
                market.atr,
                market.rsi,
                market.time_to_news if market.time_to_news is not None else 0
            ]).reshape(1, -1)
            
            # Predict probabilities for each class (expiry)
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(features)[0]
                best_class = np.argmax(probas)
                confidence = probas[best_class]
            else:
                best_class = self.model.predict(features)[0]
                confidence = 0.8  # Default confidence if no probabilities
            
            # Map class to actual expiry time
            if available_expiries is None:
                available_expiries = self.config.expiry_steps
            
            # Ensure we have a valid prediction
            if 0 <= best_class < len(available_expiries):
                return available_expiries[best_class], float(confidence)
            else:
                logger.warning(f"Invalid class predicted: {best_class}, falling back to rule-based")
                return self.get_optimal_expiry_rule_based(market, available_expiries)
                
        except Exception as e:
            logger.error(f"ML prediction failed: {e}, falling back to rule-based")
            return self.get_optimal_expiry_rule_based(market, available_expiries)
    
    def train_model(
        self,
        X: Array,
        y: Array,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the ML model on historical data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        if not self.config.use_ml:
            logger.warning("ML is disabled in config, not training")
            return {"status": "skipped", "reason": "ML disabled in config"}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Train model
            start_time = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Get feature importances if available
            feature_importances = {}
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                importances = self.model.named_steps['classifier'].feature_importances_
                feature_importances = {
                    feat: imp for feat, imp in zip(self.feature_columns, importances)
                }
            
            # Save the trained model
            self.save_model()
            
            return {
                "status": "success",
                "train_accuracy": float(train_score),
                "test_accuracy": float(test_score),
                "training_time_seconds": training_time,
                "feature_importances": feature_importances,
                "model_path": self.config.model_path
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_optimal_expiry(
        self,
        df: DataFrame,
        use_ml: Optional[bool] = None,
        available_expiries: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Get optimal expiry time based on current market conditions.
        
        Args:
            df: DataFrame with OHLCV data
            use_ml: Whether to use ML model (overrides config if not None)
            available_expiries: List of available expiry times in minutes
            
        Returns:
            Dictionary with expiry details
        """
        if use_ml is None:
            use_ml = self.config.use_ml
        
        try:
            # Calculate current market conditions
            market = self.calculate_market_conditions(df)
            
            # Get optimal expiry
            if use_ml and self.model is not None:
                expiry, confidence = self.get_optimal_expiry_ml(market, available_expiries)
                method = 'ml'
            else:
                expiry, confidence = self.get_optimal_expiry_rule_based(market, available_expiries)
                method = 'rule_based'
            
            # Ensure expiry is within bounds
            expiry = max(self.config.min_expiry, min(expiry, self.config.max_expiry))
            
            return {
                "expiry": expiry,
                "expiry_type": ExpiryType.MINUTES.value,
                "confidence": float(confidence),
                "method": method,
                "market_conditions": {
                    "volatility": market.volatility,
                    "trend_strength": market.trend_strength,
                    "volume_ratio": market.volume_ratio,
                    "atr": market.atr,
                    "rsi": market.rsi
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting optimal expiry: {e}")
            # Return default expiry on error
            return {
                "expiry": self.config.default_expiry,
                "expiry_type": ExpiryType.MINUTES.value,
                "confidence": 0.5,
                "method": "default",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = ExpiryConfig(
        min_expiry=1,
        max_expiry=60,
        default_expiry=5,
        expiry_steps=[1, 2, 5, 15, 30, 60],
        use_ml=True
    )
    
    # Initialize selector
    selector = AdvancedExpirySelector(config)
    
    # Example: Simulate some market data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5T')
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(100, 1000, 100)
    })
    df['high'] = df['open'] + np.random.rand(100) * 0.5
    df['low'] = df['open'] - np.random.rand(100) * 0.5
    df['close'] = (df['high'] + df['low']) / 2
    df.index = dates
    
    # Get optimal expiry
    result = selector.get_optimal_expiry(df)
    print("\nOptimal Expiry:")
    print(json.dumps(result, indent=2))
    
    # Example of training (would need labeled data in practice)
    print("\nTo train the model, prepare labeled data and call:")
    print("selector.train_model(X_train, y_train)")