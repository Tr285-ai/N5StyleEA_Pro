# ml_core_v15_2.py
"""
ML core for N5StyleEA_v15.2

Provides:
 - predict_direction(signal_or_df) -> float probability 0..1
 - predict_regime(signal_or_candles) -> ("trend"|"range"|"volatile", confidence)
 - estimate_volatility_cluster(...) -> float (0..1)
 - detect_liquidity_zone(...) -> "none"|"low"|"medium"|"high"
 - pick_optimal_expiry(signal, ml_prob, regime, vol_cluster) -> expiry_seconds
 - compute_final_probability(...) -> fused probability 0..1

Model training & utilities included:
 - train_direction_model(data_csv, out_model_path)
 - load_model()
 - save_model()

Notes:
 - This is intentionally practical: it includes sensible fallbacks if no trained model exists.
 - For production, train with many labeled rows (signal features + outcome).
"""

import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import logging
import gc
from datetime import datetime
from typing import Union, Tuple, Dict, Any, Optional, List
# main_v15_2.py (updated)
import asyncio
import logging
import signal
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
# In main_v15_2.py
import pandas as pd
from typing import Optional
# At the top of main_v15_2.py
import os
from dotenv import load_dotenv
# In your main_v15_2.py
async def main():
    # ... existing imports ...
    from system_monitor import SystemMonitor
    from auto_updater import AutoUpdater
 # In ml_core_v15_2.py
from optim.optimizer import PerformanceOptimizer, DataOptimizer

class TradingSystem:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.data_optimizer = DataOptimizer()
        self.performance_optimizer = PerformanceOptimizer()
        
    @PerformanceOptimizer.time_execution
    @PerformanceOptimizer.memory_usage
    async def process_market_data(self, symbol: str, data: Dict[str, Any]):
        """Process market data with performance monitoring."""
        # Optimize data before processing
        df = pd.DataFrame([data])
        df_optimized = self.data_optimizer.optimize_dataframe(df)
        
        # Process data
        signals = await self.signal_engine.generate_signals(df_optimized)
        
        # Execute trades
        for signal in signals:
            await self.strategy.execute_trade(
                symbol=symbol,
                signal=signal,
                current_price=data['close'],
                timestamp=pd.Timestamp.now()
            ) 

    # Initialize system monitor
    monitor = SystemMonitor()
    monitor_task = asyncio.create_task(monitor.start())
    
    # Initialize auto-updater
    updater = AutoUpdater()
    if updater.config['auto_install']:
        updater_task = asyncio.create_task(updater.main())
    else:
        updater_task = asyncio.create_task(updater.check_for_updates_periodically())
    
    try:
        # Your existing main loop
        trading_system = TradingSystem(config)
        await trading_system.initialize()
        await trading_system.run()
    except asyncio.CancelledError:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        monitor_task.cancel()
        updater_task.cancel()
        await asyncio.gather(
            monitor_task,
            updater_task,
            return_exceptions=True
        )
        logger.info("Shutdown complete")

if __name__ == "__main__":
    # ... existing main block ...

# Load environment variables from .env file
load_dotenv()

# Then in your config loading, you can use:
config['ml']['openai_api_key'] = os.getenv('OPENAI_API_KEY', config['ml'].get('openai_api_key', ''))

def load_historical_data(days: int = 30) -> pd.DataFrame:
    """
    Load historical market data for model training.
    
    Args:
        days: Number of days of historical data to load
        
    Returns:
        DataFrame with historical OHLCV data and technical indicators
    """
    # TODO: Implement data loading from your data source
    # This is a placeholder - replace with your actual data loading logic
    try:
        # Example: Load from CSV
        # df = pd.read_csv('data/historical_data.csv')
        # Or from database:
        # df = pd.read_sql('SELECT * FROM market_data WHERE date >= ?', 
        #                 conn, params=[pd.Timestamp.now() - pd.Timedelta(days=days)])
        
        # For now, return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'target'  # Target variable for supervised learning
        ])
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise
async def main():
    # Load configurations
    config = load_config('config/config.yaml')
    ml_config = load_config('config/ml_config.yaml')
    config['ml'] = ml_config['ml']
    
    # Create and initialize trading system
    trading_system = TradingSystem(config)
    await trading_system.initialize()
    
    # Load historical data for initial training
    logger.info("Loading historical data for initial model training...")
    try:
        historical_data = load_historical_data(days=30)
        if not historical_data.empty:
            logger.info(f"Training models with {len(historical_data)} data points...")
            await trading_system.ml_integration.train_models(historical_data)
        else:
            logger.warning("No historical data available for initial training")
    except Exception as e:
        logger.error(f"Error during initial model training: {e}")
        # Continue with default models if training fails
    
    # Start the main trading loop
    logger.info("Starting main trading loop...")
    await trading_system.run()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_system.log')
        ]
    )
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
async def main():
    # Load configurations
    config = load_config('config/config.yaml')
    ml_config = load_config('config/ml_config.yaml')
    config['ml'] = ml_config['ml']
    
    # Create and initialize trading system
    trading_system = TradingSystem(config)
    await trading_system.initialize()
    
    # Load historical data for initial training
    historical_data = load_historical_data()  # Implement this function
    await trading_system.ml_integration.train_models(historical_data)
    
    # Start the main trading loop
    await trading_system.run()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log')
    ]
)
logger = logging.getLogger('trading_system')

class TradingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.brokers = {}
        self.strategy_manager = None
        self.running = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received, stopping trading system...")
        self.running = False

    async def initialize(self):
        """Initialize the trading system."""
        logger.info("Initializing trading system...")
        
        # Initialize brokers
        from brokers_init import BrokerManager
        self.broker_manager = BrokerManager(self.config.get('brokers', {}))
        await self.broker_manager.initialize()
        
        # Initialize strategies
        from strategies.strategy_manager import StrategyManager
        self.strategy_manager = StrategyManager(self.config.get('strategies', {}))
        await self.strategy_manager.initialize()
        
        logger.info("Trading system initialized")

    async def run(self):
        """Main trading loop."""
        self.running = True
        logger.info("Starting trading system...")
        
        try:
            while self.running:
                # Main trading logic will go here
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shut down the trading system gracefully."""
        logger.info("Shutting down trading system...")
        
        # Shut down all brokers
        if hasattr(self, 'broker_manager'):
            await self.broker_manager.shutdown()
        
        logger.info("Trading system shutdown complete")

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {str(e)}")
        return {}

async def main():
    # Load configuration
    config = load_config('config/strategies.yaml')
    
    # Create and run the trading system
    trading_system = TradingSystem(config)
    await trading_system.initialize()
    await trading_system.run()

if __name__ == "__main__":
    asyncio.run(main())

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try xgboost
try:
    import xgboost as xgb
    _HAS_XGB = True
    logger.info("XGBoost successfully imported")
except ImportError as e:
    _HAS_XGB = False
    logger.warning(f"XGBoost not available, falling back to RandomForest: {e}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "direction_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "direction_scaler.pkl")
META_PATH = os.path.join(MODELS_DIR, "direction_meta.json")
EPSILON = 1e-9  # Small value to prevent division by zero

# -------------------------
# Feature builder (from candles DataFrame)
# -------------------------
def build_features_from_df(df: pd.DataFrame, lookback: int = 30) -> np.ndarray:
    """
    Build features from OHLCV data.
    
    Args:
        df: DataFrame with columns ['open','high','low','close','volume','ts'] newest at end
        lookback: Number of periods to look back for feature calculation
        
    Returns:
        np.ndarray: 1D numpy array of fixed features
    """
    try:
        if df is None or df.empty or len(df) < 3:
            logger.warning("Insufficient data for feature building")
            return np.zeros(20, dtype=float)

        # Use last lookback rows (pad if necessary)
        d = df.copy().tail(lookback).reset_index(drop=True)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in d.columns for col in required_cols):
            logger.error(f"Missing required columns. Expected: {required_cols}, found: {d.columns.tolist()}")
            return np.zeros(20, dtype=float)
            
        # Extract price data
        closes = d["close"].astype(float).values
        opens = d["open"].astype(float).values
        highs = d["high"].astype(float).values
        lows = d["low"].astype(float).values
        vols = d["volume"].astype(float).values if "volume" in d.columns else np.ones(len(d))

        # Basic returns with epsilon to prevent division by zero
        ret = np.diff(closes) / (np.abs(closes[:-1]) + EPSILON)
        avg_ret = float(np.mean(ret)) if len(ret) > 0 else 0.0
        std_ret = float(np.std(ret)) if len(ret) > 1 else 0.0

        # Momentum features
        mom1 = (closes[-1] - closes[-2]) / (np.abs(closes[-2]) + EPSILON) if len(closes) >= 2 else 0.0
        mom3 = (closes[-1] - closes[-4]) / (np.abs(closes[-4]) + EPSILON) if len(closes) >= 4 else mom1
        mom5 = (closes[-1] - closes[-6]) / (np.abs(closes[-6]) + EPSILON) if len(closes) >= 6 else mom3

        # Volatility/ATR proxy
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1))
            )
        )
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else 0.0

        # Volume features
        vol_mean = float(np.mean(vols)) if len(vols) > 0 else 0.0
        vol_std = np.std(vols) if len(vols) > 1 else 1.0
        vol_z = (vols[-1] - vol_mean) / (vol_std + EPSILON) if len(vols) > 0 else 0.0

        # RSI proxy
        up = np.where(ret > 0, ret, 0.0)
        down = np.where(ret < 0, -ret, 0.0)
        up_mean = np.mean(up) + EPSILON
        down_mean = np.mean(down) + EPSILON
        rs = up_mean / down_mean
        rsi = 100.0 - (100.0 / (1.0 + rs)) if not np.isnan(rs) else 50.0

        # Last normalized close series
        close_mean = np.mean(closes) if len(closes) > 0 else 0.0
        close_std = np.std(closes) if len(closes) > 1 else 1.0
        norm = (closes - close_mean) / (close_std + EPSILON)
        last_5 = norm[-5:] if len(norm) >= 5 else np.pad(norm, (5 - len(norm), 0), 'constant')

        # Combine all features
        feat = [avg_ret, std_ret, mom1, mom3, mom5, atr, vol_mean, vol_z, rsi]
        feat.extend(list(last_5[-5:]))
        
        # Pad to length 20 if needed
        if len(feat) < 20:
            feat = feat + [0.0] * (20 - len(feat))
            
        return np.array(feat[:20], dtype=float)  # Ensure exactly 20 features
        
    except Exception as e:
        logger.error(f"Error in build_features_from_df: {str(e)}", exc_info=True)
        return np.zeros(20, dtype=float)

# -------------------------
# Model persistence
# -------------------------
def save_model(clf: Any, 
              scaler: StandardScaler, 
              meta: Optional[Dict] = None, 
              model_path: str = MODEL_PATH, 
              scaler_path: str = SCALER_PATH, 
              meta_path: str = META_PATH) -> Tuple[str, str, str]:
    """
    Save model, scaler, and metadata to disk.
    
    Args:
        clf: Trained classifier
        scaler: Fitted StandardScaler
        meta: Dictionary of metadata
        model_path: Path to save the model
        scaler_path: Path to save the scaler
        meta_path: Path to save the metadata
        
    Returns:
        Tuple of (model_path, scaler_path, meta_path)
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)
        meta = meta or {"saved_at": datetime.utcnow().isoformat()}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Model saved successfully to {model_path}")
        return model_path, scaler_path, meta_path
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}", exc_info=True)
        raise

def load_model(model_path: str = MODEL_PATH, 
              scaler_path: str = SCALER_PATH) -> Tuple[Optional[Any], Optional[StandardScaler]]:
    """
    Load model and scaler from disk.
    
    Args:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        
    Returns:
        Tuple of (classifier, scaler) or (None, None) if loading fails
    """
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logger.info(f"Loading model from {model_path}")
            clf = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return clf, scaler
        else:
            logger.warning(f"Model files not found at {model_path} or {scaler_path}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None, None

# -------------------------
# Train direction model from CSV
# -------------------------
def train_direction_model(csv_path: str, 
                         lookback: int = 30, 
                         test_size: float = 0.2, 
                         use_xgb: bool = True) -> Dict[str, Any]:
    """
    Train a direction prediction model from CSV data.
    
    Expected CSV columns: timestamp, symbol, open, high, low, close, volume, label (1=win,0=loss)
    
    Args:
        csv_path: Path to the CSV file with training data
        lookback: Number of periods to look back for feature calculation
        test_size: Fraction of data to use for testing
        use_xgb: Whether to use XGBoost (True) or RandomForest (False)
        
    Returns:
        Dictionary with training results
    """
    try:
        logger.info(f"Loading training data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], infer_datetime_format=True)
        
        # Validate required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
            
        # Build dataset by sliding windows grouped by symbol
        X = []
        y = []
        groups = df.groupby("symbol")
        
        logger.info(f"Processing {len(groups)} symbols...")
        for symbol, g in groups:
            g_sorted = g.sort_values("timestamp").reset_index(drop=True)
            for i in range(lookback, len(g_sorted)):
                window = g_sorted.iloc[i-lookback:i]
                features = build_features_from_df(window, lookback=lookback)
                X.append(features)
                # label from row i (signal row)
                label = int(g_sorted.iloc[i].get("label", 0))
                y.append(label)
                
        if len(X) < 10:
            raise ValueError(f"Not enough training samples. Got {len(X)}, need at least 10.")
            
        X = np.vstack(X)
        y = np.array(y, dtype=int)
        
        # Handle class imbalance
        class_ratio = np.bincount(y)
        logger.info(f"Class distribution: {dict(zip(np.unique(y), class_ratio))}")
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        
        # Initialize and train model
        logger.info("Training model...")
        if _HAS_XGB and use_xgb:
            try:
                clf = xgb.XGBClassifier(
                    n_estimators=200,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    scale_pos_weight=(class_ratio[0]/class_ratio[1]) if len(class_ratio) > 1 else 1.0
                )
                logger.info("Using XGBoost classifier")
            except Exception as e:
                logger.warning(f"XGBoost initialization failed, falling back to RandomForest: {e}")
                use_xgb = False
                
        if not _HAS_XGB or not use_xgb:
            clf = RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            )
            logger.info("Using RandomForest classifier")
        
        # Split data
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, 
            y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Train model
        clf.fit(Xtr, ytr)
        
        # Evaluate
        ypred = clf.predict(Xte)
        acc = accuracy_score(yte, ypred)
        
        # Save model
        meta = {
            "trained_at": datetime.utcnow().isoformat(),
            "rows": int(len(y)),
            "accuracy": float(acc),
            "model_type": "XGBoost" if use_xgb else "RandomForest",
            "feature_count": X.shape[1],
            "class_distribution": dict(zip(np.unique(y), np.bincount(y))),
            "test_size": test_size,
            "lookback": lookback
        }
        
        model_path, scaler_path, meta_path = save_model(clf, scaler, meta)
        
        # Clean up
        del X, y, Xtr, Xte, ytr, yte
        gc.collect()
        
        logger.info(f"Training complete. Accuracy: {acc:.4f}")
        return {
            "accuracy": acc,
            "rows": meta["rows"],
            "meta": meta,
            "model_path": model_path
        }
        
    except Exception as e:
        logger.error(f"Error in train_direction_model: {str(e)}", exc_info=True)
        raise
  def predict_direction(signal_or_df: Union[Dict[str, Any], pd.DataFrame]) -> float:
    """
    Predict market direction using trained model or fallback to heuristic.
    
    Args:
        signal_or_df: Either a signal dictionary or a pandas DataFrame with OHLCV data
        
    Returns:
        float: Probability between 0 and 1 (1.0 = strong buy, 0.0 = strong sell)
    """
    try:
        # Try to load model
        clf, scaler = load_model()
        
        # Prepare features based on input type
        if isinstance(signal_or_df, dict):
            signal = signal_or_df
            # Try to create df from signal["history"] if provided
            if "history" in signal and isinstance(signal["history"], list):
                try:
                    df = pd.DataFrame(
                        signal["history"], 
                        columns=["open", "high", "low", "close", "volume", "ts"]
                    )
                except Exception as e:
                    logger.warning(f"Could not create DataFrame from history: {e}")
                    df = None
            else:
                df = None
        else:
            df = signal_or_df

        # Use ML model if available and data is suitable
        if clf is not None and scaler is not None and df is not None:
            try:
                feats = build_features_from_df(df)
                if feats is not None and not np.isnan(feats).any():
                    feats_s = scaler.transform(feats.reshape(1, -1))
                    proba = float(clf.predict_proba(feats_s)[0, 1])
                    return max(0.0, min(1.0, proba))
            except Exception as e:
                logger.warning(f"ML prediction failed, falling back to heuristic: {e}")

        # Fallback heuristic
        logger.debug("Using fallback heuristic for direction prediction")
        try:
            if isinstance(signal_or_df, dict):
                s = signal_or_df
                close = float(s.get("price", 0.0))
                prev = float(s.get("price_prev", close))
                prev2 = float(s.get("price_prev2", prev))
            else:
                closes = signal_or_df["close"].values
                close = float(closes[-1]) if len(closes) > 0 else 0.0
                prev = float(closes[-2]) if len(closes) > 1 else close
                prev2 = float(closes[-3]) if len(closes) > 2 else prev

            # Simple momentum-based heuristic
            mom1 = (close - prev) / (prev + EPSILON)
            mom2 = (prev - prev2) / (prev2 + EPSILON)
            rsi_like = 1.0 / (1.0 + math.exp(-10.0 * (mom1 - 0.5 * mom2)))
            return float(np.clip(rsi_like, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error in fallback heuristic: {e}", exc_info=True)
            return 0.5  # Neutral prediction on error
            
    except Exception as e:
        logger.error(f"Error in predict_direction: {e}", exc_info=True)
        return 0.5  # Fallback to neutral prediction

def predict_regime(signal_or_candles: Union[Dict, pd.DataFrame], 
                  lookback: int = 50) -> Tuple[str, float]:
    """
    Predict market regime: "trend", "range", or "volatile".
    
    Args:
        signal_or_candles: Either a signal dict or DataFrame with OHLCV data
        lookback: Number of periods to analyze
        
    Returns:
        Tuple of (regime, confidence) where:
        - regime: "trend"|"range"|"volatile"
        - confidence: float between 0 and 1
    """
    try:
        # Extract price data
        if isinstance(signal_or_candles, dict):
            # Try to get candles from signal
            if "history" in signal_or_candles and isinstance(signal_or_candles["history"], list):
                try:
                    df = pd.DataFrame(
                        signal_or_candles["history"],
                        columns=["open", "high", "low", "close", "volume", "ts"]
                    )
                    closes = df["close"].values
                except:
                    logger.warning("Could not extract price data from signal")
                    return "range", 0.5
            else:
                logger.warning("No price history in signal")
                return "range", 0.5
        else:
            df = signal_or_candles
            closes = df["close"].values if "close" in df.columns else None

        if closes is None or len(closes) < lookback:
            logger.warning("Insufficient data for regime detection")
            return "range", 0.5

        # Use last lookback periods
        prices = closes[-lookback:]
        returns = np.diff(np.log(prices + EPSILON))
        
        if len(returns) < 2:
            return "range", 0.5

        # Calculate metrics
        vol = np.std(returns)
        abs_returns = np.abs(returns)
        autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        
        # Trend detection (ADX-like)
        high = df["high"].values[-lookback:] if "high" in df.columns else prices
        low = df["low"].values[-lookback:] if "low" in df.columns else prices
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - prices[:-1]),
                np.abs(low[1:] - prices[:-1])
            )
        )
        atr = np.mean(tr) if len(tr) > 0 else 0.0
        directional_move = np.abs(prices[-1] - prices[0])
        adx_like = 100.0 * (directional_move / (atr * len(prices) + EPSILON))
        
        # Determine regime
        if adx_like > 25.0:
            return "trend", min(1.0, adx_like / 50.0)
        elif vol > np.percentile(abs_returns, 75) * 1.5:
            return "volatile", min(1.0, vol * 10.0)
        else:
            return "range", 0.7
            
    except Exception as e:
        logger.error(f"Error in predict_regime: {e}", exc_info=True)
        return "range", 0.5

def estimate_volatility_cluster(signal_or_candles: Union[Dict, pd.DataFrame], 
                               lookback: int = 20) -> float:
    """
    Estimate volatility clustering (burstiness) in returns.
    
    Args:
        signal_or_candles: Either a signal dict or DataFrame with OHLCV data
        lookback: Number of periods to analyze
        
    Returns:
        float: Value between 0 (no clustering) and 1 (strong clustering)
    """
    try:
        # Extract price data
        if isinstance(signal_or_candles, dict):
            # Try to get candles from signal
            if "history" in signal_or_candles and isinstance(signal_or_candles["history"], list):
                try:
                    df = pd.DataFrame(
                        signal_or_candles["history"],
                        columns=["open", "high", "low", "close", "volume", "ts"]
                    )
                    closes = df["close"].values
                except:
                    logger.warning("Could not extract price data from signal")
                    return 0.5
            else:
                logger.warning("No price history in signal")
                return 0.5
        else:
            df = signal_or_candles
            closes = df["close"].values if "close" in df.columns else None

        if closes is None or len(closes) < lookback:
            logger.warning("Insufficient data for volatility clustering")
            return 0.5

        # Use last lookback periods
        prices = closes[-lookback:]
        returns = np.diff(np.log(prices + EPSILON))
        
        if len(returns) < 10:  # Need enough data for meaningful calculation
            return 0.5

        # Calculate volatility clustering using autocorrelation of absolute returns
        abs_returns = np.abs(returns)
        autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        
        # Normalize to 0-1 range (typical values are between 0 and 0.3)
        volatility_cluster = max(0.0, min(1.0, (autocorr + 0.1) * 2.5))
        
        return float(volatility_cluster)
        
    except Exception as e:
        logger.error(f"Error in estimate_volatility_cluster: {e}", exc_info=True)
        return 0.5

def detect_liquidity_zone(signal_or_candles: Union[Dict, pd.DataFrame],
                         lookback: int = 100) -> str:
    """
    Detect liquidity zones based on volume profile.
    
    Args:
        signal_or_candles: Either a signal dict or DataFrame with OHLCV data
        lookback: Number of periods to analyze
        
    Returns:
        str: "none"|"low"|"medium"|"high" indicating liquidity level
    """
    try:
        # Extract volume data
        if isinstance(signal_or_candles, dict):
            # Try to get volume from signal
            if "history" in signal_or_candles and isinstance(signal_or_candles["history"], list):
                try:
                    df = pd.DataFrame(
                        signal_or_candles["history"],
                        columns=["open", "high", "low", "close", "volume", "ts"]
                    )
                    volumes = df["volume"].values if "volume" in df.columns else None
                except:
                    logger.warning("Could not extract volume data from signal")
                    return "none"
            else:
                logger.warning("No price history in signal")
                return "none"
        else:
            volumes = signal_or_candles["volume"].values if "volume" in signal_or_candles.columns else None

        if volumes is None or len(volumes) < lookback:
            logger.warning("Insufficient data for liquidity detection")
            return "none"

        # Use last lookback periods
        recent_volumes = volumes[-lookback:]
        
        # Calculate volume statistics
        vol_mean = np.mean(recent_volumes)
        vol_std = np.std(recent_volumes) if len(recent_volumes) > 1 else 0.0
        current_vol = recent_volumes[-1] if len(recent_volumes) > 0 else 0.0
        
        # Determine liquidity level
        if vol_std < EPSILON:
            return "medium"  # Can't determine, return medium as default
            
        z_score = (current_vol - vol_mean) / (vol_std + EPSILON)
        
        if z_score > 2.0:
            return "high"
        elif z_score > 0.5:
            return "medium"
        elif z_score > -0.5:
            return "low"
        else:
            return "none"
            
    except Exception as e:
        logger.error(f"Error in detect_liquidity_zone: {e}", exc_info=True)
        return "none"

def pick_optimal_expiry(signal: Dict, 
                       ml_prob: float, 
                       regime: str, 
                       vol_cluster: float,
                       min_expiry: int = 30,
                       max_expiry: int = 300) -> int:
    """
    Pick optimal expiry time based on market conditions.
    
    Args:
        signal: Signal dictionary
        ml_prob: ML probability (0-1)
        regime: Market regime ("trend", "range", or "volatile")
        vol_cluster: Volatility clustering score (0-1)
        min_expiry: Minimum expiry in seconds
        max_expiry: Maximum expiry in seconds
        
    Returns:
        int: Expiry time in seconds
    """
    try:
        # Base expiry based on regime
        if regime == "trend":
            base_expiry = min_expiry + int(0.6 * (max_expiry - min_expiry))
        elif regime == "volatile":
            base_expiry = min_expiry + int(0.3 * (max_expiry - min_expiry))
        else:  # range
            base_expiry = min_expiry + int(0.4 * (max_expiry - min_expiry))
            
        # Adjust for ML confidence
        confidence = 2.0 * abs(ml_prob - 0.5)  # 0 for 0.5, 1 for 0 or 1
        expiry = int(base_expiry * (1.0 - 0.3 * confidence))  # Shorter expiry for high confidence
        
        # Adjust for volatility clustering
        expiry = int(expiry * (1.0 - 0.4 * vol_cluster))  # Shorter expiry in high vol clusters
        
        # Ensure within bounds
        return max(min_expiry, min(max_expiry, expiry))
        
    except Exception as e:
        logger.error(f"Error in pick_optimal_expiry: {e}", exc_info=True)
        return (min_expiry + max_expiry) // 2  # Return midpoint on error

def compute_final_probability(ml_prob: float,
                            regime: str,
                            vol_cluster: float,
                            atr: float,
                            rsi: float) -> float:
    """
    Fuse multiple signals into a final probability.
    
    Args:
        ml_prob: ML model probability (0-1)
        regime: Market regime ("trend", "range", or "volatile")
        vol_cluster: Volatility clustering score (0-1)
        atr: Average True Range (normalized)
        rsi: RSI value (0-100)
        
    Returns:
        float: Fused probability (0-1)
    """
    try:
        # Base probability from ML model
        prob = ml_prob
        
        # Adjust based on regime
        if regime == "trend":
            # In trends, be more confident in the direction
            prob = 0.5 + 0.5 * (prob - 0.5) * 1.5
        elif regime == "range":
            # In ranges, be less confident
            prob = 0.5 + 0.8 * (prob - 0.5)
            
        # Adjust for volatility clustering
        # Higher vol clustering -> mean reversion more likely
        if prob > 0.5:
            prob = prob * (1.0 - 0.2 * vol_cluster) + 0.5 * 0.2 * vol_cluster
        else:
            prob = prob * (1.0 - 0.2 * vol_cluster) + 0.5 * 0.2 * vol_cluster
            
        # Adjust for RSI (mean reversion)
        rsi_factor = min(abs(rsi - 50) / 30.0, 1.0)  # 0 at 50, 1 at <20 or >80
        if (rsi > 70 and prob > 0.5) or (rsi < 30 and prob < 0.5):
            # Overbought/oversold and going with the trend -> reduce confidence
            prob = 0.5 + 0.8 * (prob - 0.5)
            
        # Ensure within bounds
        return max(0.01, min(0.99, prob))
        
    except Exception as e:
        logger.error(f"Error in compute_final_probability: {e}", exc_info=True)
        return 0.5  # Neutral on error

if __name__ == "__main__":
    # Example usage
    try:
        # Example signal
        example_signal = {
            "price": 100.0,
            "price_prev": 99.5,
            "price_prev2": 99.0,
            "history": [
                [99.0, 99.5, 98.8, 99.2, 1000, 1234567890],
                [99.2, 99.8, 99.1, 99.7, 1200, 1234567800],
                [99.7, 100.2, 99.5, 100.0, 1500, 1234567900]
            ]
        }
        
        # Make predictions
        direction_prob = predict_direction(example_signal)
        regime, regime_conf = predict_regime(example_signal)
        vol_cluster = estimate_volatility_cluster(example_signal)
        liquidity = detect_liquidity_zone(example_signal)
        
        print(f"Direction probability: {direction_prob:.2f}")
        print(f"Regime: {regime} (confidence: {regime_conf:.2f})")
        print(f"Volatility clustering: {vol_cluster:.2f}")
        print(f"Liquidity: {liquidity}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)      

# [Rest of the functions would be similarly updated...]
# Note: The remaining functions would follow the same pattern of improvements