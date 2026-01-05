# model_manager.py
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import joblib
import boto3
from botocore.exceptions import ClientError
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(
        self,
        model_dir: str = 'models',
        cloud_storage: Optional[Dict] = None,
        retrain_interval: int = 7,  # days
        min_retrain_samples: int = 1000,
        validation_split: float = 0.2
    ):
        self.model_dir = model_dir
        self.cloud_storage = cloud_storage
        self.retrain_interval = timedelta(days=retrain_interval)
        self.min_retrain_samples = min_retrain_samples
        self.validation_split = validation_split
        self.last_retrain = None
        self.current_version = None
        self.scaler = None
        self.model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load or initialize model
        self._load_latest_model()
        
    def _load_latest_model(self) -> None:
        """Load the most recent model and scaler."""
        try:
            model_path = os.path.join(self.model_dir, 'latest_model.h5')
            scaler_path = os.path.join(self.model_dir, 'latest_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.current_version = self._get_model_version()
                logger.info(f"Loaded model version {self.current_version}")
            else:
                logger.warning("No existing model found, will train a new one")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
    
    async def maybe_retrain(
        self,
        new_data: pd.DataFrame,
        model_params: Optional[Dict] = None
    ) -> bool:
        """Check if retraining is needed and perform it if necessary."""
        if not self._should_retrain(new_data):
            return False
            
        logger.info("Starting model retraining...")
        
        try:
            # Prepare data
            X, y = self._prepare_data(new_data)
            
            # Train new model
            new_model, new_scaler = await self._train_model(X, y, model_params)
            
            # Evaluate new model
            if self._evaluate_model(new_model, X, y):
                self._update_models(new_model, new_scaler)
                self._save_model_artifacts()
                self._cleanup_old_models()
                self.last_retrain = datetime.utcnow()
                self._notify_model_update()
                return True
                
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            
        return False
    
    def _should_retrain(self, new_data: pd.DataFrame) -> bool:
        """Determine if the model should be retrained."""
        if len(new_data) < self.min_retrain_samples:
            logger.info(f"Not enough new data for retraining: {len(new_data)} < {self.min_retrain_samples}")
            return False
            
        if self.last_retrain and (datetime.utcnow() - self.last_retrain) < self.retrain_interval:
            logger.info("Not enough time passed since last retraining")
            return False
            
        return True
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training."""
        # Feature engineering and scaling
        features = self._extract_features(data)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features)
        y = data['target'].values  # Assuming 'target' column exists
        return X, y
    
    async def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict] = None
    ) -> Tuple[tf.keras.Model, StandardScaler]:
        """Train a new model with the given data and parameters."""
        # Use provided parameters or defaults
        params = params or {
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'patience': 10
        }
        
        # Create or clone model
        if self.model:
            new_model = clone_model(self.model)
            new_model.set_weights(self.model.get_weights())
        else:
            new_model = self._create_model(X.shape[1], params)
            
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        new_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=params['patience'],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'temp_model.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train model
        history = await new_model.fit(
            X, y,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best weights
        new_model = load_model(os.path.join(self.model_dir, 'temp_model.h5'))
        os.remove(os.path.join(self.model_dir, 'temp_model.h5'))
        
        return new_model, self.scaler
    
    def _evaluate_model(
        self,
        model: tf.keras.Model,
        X: np.ndarray,
        y: np.ndarray
    ) -> bool:
        """Evaluate if the new model is better than the current one."""
        if not self.model:
            return True
            
        # Get predictions
        current_preds = self.model.predict(X)
        new_preds = model.predict(X)
        
        # Simple accuracy comparison (can be enhanced with more metrics)
        current_acc = np.mean((current_preds > 0.5) == y)
        new_acc = np.mean((new_preds > 0.5) == y)
        
        logger.info(f"Model evaluation - Current: {current_acc:.4f}, New: {new_acc:.4f}")
        return new_acc > current_acc * 1.01  # At least 1% improvement
    
    def _update_models(
        self,
        new_model: tf.keras.Model,
        new_scaler: StandardScaler
    ) -> None:
        """Update the current model and scaler."""
        old_model = self.model
        old_scaler = self.scaler
        
        try:
            self.model = new_model
            self.scaler = new_scaler
            self.current_version = self._get_next_version()
            logger.info(f"Model updated to version {self.current_version}")
        except Exception as e:
            # Revert on error
            self.model = old_model
            self.scaler = old_scaler
            raise e
    
    def _save_model_artifacts(self) -> None:
        """Save model and metadata."""
        if not self.model or not self.scaler:
            return
            
        # Save model and scaler
        model_path = os.path.join(self.model_dir, 'latest_model.h5')
        scaler_path = os.path.join(self.model_dir, 'latest_scaler.pkl')
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save versioned copies
        versioned_model = os.path.join(
            self.model_dir,
            f'model_v{self.current_version}.h5'
        )
        versioned_scaler = os.path.join(
            self.model_dir,
            f'scaler_v{self.current_version}.pkl'
        )
        self.model.save(versioned_model)
        joblib.dump(self.scaler, versioned_scaler)
        
        # Save metadata
        metadata = {
            'version': self.current_version,
            'timestamp': datetime.utcnow().isoformat(),
            'model_summary': []
        }
        
        # Get model summary
        with open(os.path.join(self.model_dir, 'model_architecture.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            self.model.summary(print_fn=lambda x: metadata['model_summary'].append(x))
        
        with open(os.path.join(self.model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Upload to cloud storage if configured
        if self.cloud_storage:
            self._upload_to_cloud(
                [model_path, scaler_path, versioned_model, versioned_scaler]
            )
    
    def _upload_to_cloud(self, files: list) -> None:
        """Upload files to cloud storage."""
        if not self.cloud_storage:
            return
            
        try:
            session = boto3.Session(
                aws_access_key_id=self.cloud_storage.get('access_key'),
                aws_secret_access_key=self.cloud_storage.get('secret_key'),
                region_name=self.cloud_storage.get('region', 'us-east-1')
            )
            s3 = session.client('s3')
            
            for file_path in files:
                if not os.path.exists(file_path):
                    continue
                    
                s3.upload_file(
                    file_path,
                    self.cloud_storage['bucket'],
                    os.path.join('models', os.path.basename(file_path))
                )
                
            logger.info("Successfully uploaded model artifacts to cloud storage")
            
        except ClientError as e:
            logger.error(f"Error uploading to cloud storage: {e}")
    
    def _cleanup_old_models(self, keep_last: int = 5) -> None:
        """Remove old model versions, keeping only the most recent ones."""
        try:
            # Get all versioned models
            model_files = [
                f for f in os.listdir(self.model_dir)
                if f.startswith('model_v') and f.endswith('.h5')
            ]
            
            # Sort by version number (descending)
            model_files.sort(
                key=lambda x: int(x.split('_v')[1].split('.h5')[0]),
                reverse=True
            )
            
            # Delete old versions
            for old_model in model_files[keep_last:]:
                try:
                    os.remove(os.path.join(self.model_dir, old_model))
                    # Also remove corresponding scaler
                    scaler = old_model.replace('model_', 'scaler_')
                    if os.path.exists(os.path.join(self.model_dir, scaler)):
                        os.remove(os.path.join(self.model_dir, scaler))
                except Exception as e:
                    logger.warning(f"Could not remove old model {old_model}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
    
    def _get_model_version(self) -> int:
        """Get the current model version number."""
        if not os.path.exists(os.path.join(self.model_dir, 'metadata.json')):
            return 1
            
        try:
            with open(os.path.join(self.model_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                return int(metadata.get('version', 0))
        except Exception:
            return 1
    
    def _get_next_version(self) -> int:
        """Get the next version number."""
        return self._get_model_version() + 1
    
    def _notify_model_update(self) -> None:
        """Notify about the model update."""
        # Could be extended to send email/slack notification
        logger.info(f"Model successfully updated to version {self.current_version}")
    
    def _create_model(self, input_dim: int, params: Dict) -> tf.keras.Model:
        """Create a new LSTM model."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                params['lstm_units'],
                input_shape=(None, input_dim),
                return_sequences=True
            ),
            tf.keras.layers.Dropout(params['dropout_rate']),
            tf.keras.layers.LSTM(params['lstm_units']),
            tf.keras.layers.Dropout(params['dropout_rate']),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model