"""
Micro Dataset Preparation v15.2

A comprehensive data preparation pipeline for training micro-prediction models.
Handles data loading, cleaning, feature engineering, and dataset splitting.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import yfinance as yf
import pickle
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prepare_micro_dataset.log')
    ]
)
logger = logging.getLogger('dataset_prep')

# Type aliases
Array = np.ndarray
DataFrame = pd.DataFrame

class TimeFrame(Enum):
    """Supported timeframes for data aggregation."""
    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'

class FeatureType(Enum):
    """Types of features to generate."""
    PRICE = 'price'            # Raw price data
    VOLUME = 'volume'          # Volume data
    VOLATILITY = 'volatility'  # Volatility measures
    MOMENTUM = 'momentum'      # Momentum indicators
    TREND = 'trend'           # Trend indicators
    OSCILLATOR = 'oscillator'  # Oscillator indicators
    CUSTOM = 'custom'         # Custom features

@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    feature_type: FeatureType
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    symbol: str
    timeframes: List[TimeFrame]
    start_date: str
    end_date: Optional[str] = None
    sequence_length: int = 60
    prediction_horizon: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    output_dir: str = "data/micro_datasets"
    features: List[FeatureConfig] = field(default_factory=list)
    target_lookahead: int = 5
    target_threshold: float = 0.0005  # 0.05% price movement
    min_sequence_length: int = 20
    max_sequence_length: int = 200
    normalize_features: bool = True
    scale_targets: bool = False
    shuffle_sequences: bool = True
    balance_classes: bool = True
    augment_data: bool = True
    save_raw_data: bool = True
    save_processed_data: bool = True
    save_metadata: bool = True

class MicroDataset:
    """
    Handles the preparation of micro-prediction datasets.
    Supports multiple timeframes, feature engineering, and dataset splitting.
    """
    
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        Initialize the dataset preparation pipeline.
        
        Args:
            config: Either a configuration dictionary or path to a JSON config file
        """
        self.config = self._load_config(config) if config else DatasetConfig(
            symbol="BTC-USD",
            timeframes=[TimeFrame.M5],
            start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        # Initialize data storage
        self.raw_data: Dict[TimeFrame, DataFrame] = {}
        self.processed_data: Dict[TimeFrame, Dict[str, Any]] = {}
        self.feature_scalers: Dict[str, Any] = {}
        self.target_scaler: Optional[Any] = None
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'statistics': {}
        }
        
        # Create output directories
        self.output_dir = Path(self.config.output_dir)
        self.raw_dir = self.output_dir / 'raw'
        self.processed_dir = self.output_dir / 'processed'
        self.models_dir = self.output_dir / 'models'
        
        for d in [self.raw_dir, self.processed_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info("MicroDataset initialized")
    
    def _load_config(self, config: Union[Dict, str]) -> DatasetConfig:
        """Load configuration from dict or file."""
        if isinstance(config, str):
            with open(config, 'r') as f:
                config_data = json.load(f)
        else:
            config_data = config
        
        # Convert timeframes to enum
        if 'timeframes' in config_data:
            config_data['timeframes'] = [
                TimeFrame(tf) if isinstance(tf, str) else tf
                for tf in config_data['timeframes']
            ]
        
        # Convert feature configs
        if 'features' in config_data:
            config_data['features'] = [
                FeatureConfig(
                    feature_type=FeatureType(feat.get('feature_type')),
                    params=feat.get('params', {}),
                    enabled=feat.get('enabled', True)
                )
                for feat in config_data['features']
            ]
        
        return DatasetConfig(**config_data)
    
    def fetch_data(self) -> None:
        """Fetch raw market data from the specified source."""
        logger.info(f"Fetching data for {self.config.symbol}")
        
        for tf in self.config.timeframes:
            try:
                logger.info(f"Downloading {tf.value} data from {self.config.start_date} to {self.config.end_date}")
                
                # Use yfinance to download data
                data = yf.download(
                    self.config.symbol,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=tf.value,
                    progress=False
                )
                
                if data.empty:
                    logger.warning(f"No data returned for {tf.value} timeframe")
                    continue
                
                # Ensure proper datetime index
                data.index = pd.to_datetime(data.index)
                
                # Store raw data
                self.raw_data[tf] = data
                
                # Save raw data
                if self.config.save_raw_data:
                    filename = f"{self.config.symbol.replace('-', '_')}_{tf.value}_raw.csv"
                    filepath = self.raw_dir / filename
                    data.to_csv(filepath)
                    logger.info(f"Saved raw {tf.value} data to {filepath}")
                
                # Log basic statistics
                self.metadata['statistics'][f"{tf.value}_raw"] = {
                    'start_date': data.index[0].isoformat(),
                    'end_date': data.index[-1].isoformat(),
                    'num_bars': len(data),
                    'ohlc_stats': {
                        'open': {
                            'min': float(data['Open'].min()),
                            'max': float(data['Open'].max()),
                            'mean': float(data['Open'].mean())
                        },
                        'close': {
                            'min': float(data['Close'].min()),
                            'max': float(data['Close'].max()),
                            'mean': float(data['Close'].mean())
                        },
                        'volume': {
                            'min': float(data['Volume'].min()),
                            'max': float(data['Volume'].max()),
                            'mean': float(data['Volume'].mean())
                        }
                    }
                }
                
            except Exception as e:
                logger.error(f"Error fetching {tf.value} data: {e}")
                continue
    
    def _generate_features(self, data: DataFrame, timeframe: TimeFrame) -> Tuple[DataFrame, List[str]]:
        """Generate features from raw OHLCV data."""
        df = data.copy()
        feature_columns = []
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Rename columns to lowercase for consistency
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        feature_columns.extend(['returns', 'log_returns'])
        
        # Add price movements
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        feature_columns.extend(['price_change', 'price_change_pct'])
        
        # Add volatility features
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['atr'] = talib.ATR(
            df['high'], df['low'], df['close'], 
            timeperiod=14
        )
        feature_columns.extend(['volatility', 'atr'])
        
        # Add momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['adx'] = talib.ADX(
            df['high'], df['low'], df['close'], 
            timeperiod=14
        )
        feature_columns.extend(['rsi', 'macd', 'macd_signal', 'macd_hist', 'adx'])
        
        # Add trend indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        feature_columns.extend(['sma_20', 'sma_50', 'ema_12', 'ema_26'])
        
        # Add Bollinger Bands
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(
            df['close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2
        )
        feature_columns.extend(['upper_bb', 'middle_bb', 'lower_bb'])
        
        # Add volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        feature_columns.extend(['volume_change', 'volume_ma', 'volume_ratio'])
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        feature_columns.extend(['hour', 'day_of_week', 'day_of_month', 'month'])
        
        # Add custom features from config
        for feature_cfg in self.config.features:
            if not feature_cfg.enabled:
                continue
                
            try:
                if feature_cfg.feature_type == FeatureType.CUSTOM:
                    # Handle custom feature generation
                    pass
                # Add more feature types as needed
                    
            except Exception as e:
                logger.warning(f"Failed to generate feature {feature_cfg.feature_type}: {e}")
        
        # Clean up any infinite or NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        return df, feature_columns
    
    def _create_sequences(
        self, 
        data: DataFrame,
        feature_columns: List[str],
        target_column: str = 'target'
    ) -> Tuple[Array, Array, List[datetime]]:
        """Create input sequences and targets for training."""
        X, y, timestamps = [], [], []
        data_values = data[feature_columns].values
        
        # Create sequences
        for i in range(len(data) - self.config.sequence_length - self.config.prediction_horizon):
            # Input sequence
            seq = data_values[i:(i + self.config.sequence_length)]
            
            # Target (future price movement)
            future_prices = data['close'].iloc[
                i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_horizon
            ].values
            
            if len(future_prices) < self.config.prediction_horizon:
                continue
                
            # Simple binary target: 1 if price goes up, 0 otherwise
            price_change = (future_prices[-1] - data['close'].iloc[i + self.config.sequence_length - 1]) / \
                          data['close'].iloc[i + self.config.sequence_length - 1]
            
            target = 1 if price_change > self.config.target_threshold else (
                0 if price_change < -self.config.target_threshold else 2
            )  # 0: down, 1: up, 2: neutral
            
            X.append(seq)
            y.append(target)
            timestamps.append(data.index[i + self.config.sequence_length - 1])
        
        return np.array(X), np.array(y), timestamps
    
    def prepare_dataset(self) -> None:
        """Prepare the dataset by generating features and creating sequences."""
        if not self.raw_data:
            logger.warning("No raw data available. Fetching data first...")
            self.fetch_data()
        
        for tf, data in self.raw_data.items():
            try:
                logger.info(f"Preparing {tf.value} dataset...")
                
                # Generate features
                df, feature_columns = self._generate_features(data, tf)
                
                # Create sequences
                X, y, timestamps = self._create_sequences(df, feature_columns)
                
                if len(X) == 0:
                    logger.warning(f"No valid sequences created for {tf.value}")
                    continue
                
                # Split into train/validation/test sets
                X_train, X_temp, y_train, y_temp, ts_train, ts_temp = train_test_split(
                    X, y, timestamps,
                    test_size=(1 - self.config.train_ratio),
                    random_state=self.config.random_seed,
                    stratify=y if self.config.balance_classes else None
                )
                
                val_ratio = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
                X_val, X_test, y_val, y_test, ts_val, ts_test = train_test_split(
                    X_temp, y_temp, ts_temp,
                    test_size=(1 - val_ratio),
                    random_state=self.config.random_seed,
                    stratify=y_temp if self.config.balance_classes else None
                )
                
                # Normalize features
                if self.config.normalize_features:
                    self.feature_scalers[str(tf.value)] = StandardScaler()
                    X_train = self.feature_scalers[str(tf.value)].fit_transform(
                        X_train.reshape(-1, X_train.shape[-1])
                    ).reshape(X_train.shape)
                    
                    X_val = self.feature_scalers[str(tf.value)].transform(
                        X_val.reshape(-1, X_val.shape[-1])
                    ).reshape(X_val.shape)
                    
                    X_test = self.feature_scalers[str(tf.value)].transform(
                        X_test.reshape(-1, X_test.shape[-1])
                    ).reshape(X_test.shape)
                
                # Save processed data
                self.processed_data[tf] = {
                    'X_train': X_train,
                    'X_val': X_val,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test,
                    'timestamps_train': ts_train,
                    'timestamps_val': ts_val,
                    'timestamps_test': ts_test,
                    'feature_columns': feature_columns
                }
                
                # Update metadata
                self.metadata['statistics'][f"{tf.value}_processed"] = {
                    'num_samples': {
                        'train': len(X_train),
                        'validation': len(X_val),
                        'test': len(X_test)
                    },
                    'class_distribution': {
                        'train': {
                            'down': int((y_train == 0).sum()),
                            'up': int((y_train == 1).sum()),
                            'neutral': int((y_train == 2).sum())
                        },
                        'validation': {
                            'down': int((y_val == 0).sum()),
                            'up': int((y_val == 1).sum()),
                            'neutral': int((y_val == 2).sum())
                        },
                        'test': {
                            'down': int((y_test == 0).sum()),
                            'up': int((y_test == 1).sum()),
                            'neutral': int((y_test == 2).sum())
                        }
                    },
                    'feature_columns': feature_columns
                }
                
                logger.info(f"Prepared {tf.value} dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
                
            except Exception as e:
                logger.error(f"Error preparing {tf.value} dataset: {e}")
                continue
        
        # Save processed data and metadata
        if self.config.save_processed_data:
            self.save_processed_data()
        
        logger.info("Dataset preparation completed")
    
    def save_processed_data(self) -> None:
        """Save processed datasets and metadata."""
        if not self.processed_data:
            logger.warning("No processed data to save")
            return
        
        # Save processed data for each timeframe
        for tf, data in self.processed_data.items():
            try:
                # Create output directory for this timeframe
                tf_dir = self.processed_dir / str(tf.value)
                tf_dir.mkdir(exist_ok=True)
                
                # Save datasets
                for split in ['train', 'val', 'test']:
                    np.save(tf_dir / f'X_{split}.npy', data[f'X_{split}'])
                    np.save(tf_dir / f'y_{split}.npy', data[f'y_{split}'])
                    np.save(tf_dir / f'timestamps_{split}.npy', data[f'timestamps_{split}'])
                
                # Save feature columns
                with open(tf_dir / 'feature_columns.pkl', 'wb') as f:
                    pickle.dump(data['feature_columns'], f)
                
                logger.info(f"Saved processed {tf.value} data to {tf_dir}")
                
            except Exception as e:
                logger.error(f"Error saving {tf.value} processed data: {e}")
        
        # Save metadata
        if self.config.save_metadata:
            metadata_path = self.processed_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {metadata_path}")
        
        # Save feature scalers
        scaler_path = self.models_dir / 'feature_scalers.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scalers, f)
        logger.info(f"Saved feature scalers to {scaler_path}")
    
    def get_dataset(self, timeframe: Union[str, TimeFrame], split: str = 'train') -> Tuple[Array, Array]:
        """
        Get dataset for a specific timeframe and split.
        
        Args:
            timeframe: Timeframe to get data for
            split: One of 'train', 'val', or 'test'
            
        Returns:
            Tuple of (X, y) arrays
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
            
        if timeframe not in self.processed_data:
            raise ValueError(f"No processed data available for {timeframe.value}")
            
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
            
        return (
            self.processed_data[timeframe][f'X_{split}'],
            self.processed_data[timeframe][f'y_{split}']
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare micro-prediction dataset')
    parser.add_argument('--config', type=str, default='config/dataset_config.json',
                        help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Trading symbol (overrides config)')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes to process (overrides config)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD, overrides config)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD, overrides config)')
    parser.add_argument('--output-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--no-download', action='store_true', help='Skip downloading new data')
    return parser.parse_args()

def main():
    """Main function for command line usage."""
    args = parse_args()
    
    try:
        # Load config if provided
        config = {}
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Override config with command line args
        if args.symbol:
            config['symbol'] = args.symbol
        if args.timeframes:
            config['timeframes'] = args.timeframes
        if args.start_date:
            config['start_date'] = args.start_date
        if args.end_date:
            config['end_date'] = args.end_date
        if args.output_dir:
            config['output_dir'] = args.output_dir
        
        # Initialize dataset
        dataset = MicroDataset(config)
        
        # Download data if needed
        if not args.no_download:
            dataset.fetch_data()
        
        # Prepare dataset
        dataset.prepare_dataset()
        
        logger.info("Dataset preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in dataset preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()