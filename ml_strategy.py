import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from .base_strategy import BaseStrategy
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

class MLStrategy(BaseStrategy):
    def __init__(self, model_path=None, scaler_path=None, 
                 seq_length=30, prediction_threshold=0.6):
        super().__init__()
        self.model = None
        self.scaler = None
        self.seq_length = seq_length
        self.prediction_threshold = prediction_threshold
        self.name = "MLStrategy"
        
        # Load model and scaler if paths are provided
        if model_path and scaler_path:
            self.load_model(model_path, scaler_path)
            
    def load_model(self, model_path, scaler_path):
        """Load pre-trained model and scaler"""
        try:
            if model_path.endswith('.h5'):
                self.model = load_model(model_path)
            else:
                self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            
    def prepare_features(self, df):
        """Prepare features for the ML model"""
        # Technical indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI
        rsi = RSIIndicator(close=df['close'])
        df['rsi'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], 
                              close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features
        features = ['returns', 'volatility', 'rsi', 'macd', 'macd_signal', 'atr']
        X = df[features].values
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
            
        return X, df.index
    
    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM model"""
        X = []
        for i in range(len(data) - seq_length + 1):
            X.append(data[i:(i + seq_length)])
        return np.array(X)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0, index=df.index)
            
        # Prepare features
        X, valid_index = self.prepare_features(df.copy())
        
        # Create sequences for LSTM
        if len(X) >= self.seq_length:
            X_seq = self.create_sequences(X, self.seq_length)
            
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                preds = self.model.predict_proba(X_seq[-1].reshape(1, -1))[:, 1]
            else:
                preds = self.model.predict(X_seq[-1].reshape(1, self.seq_length, -1))[0][0]
                
            # Generate signals based on predictions
            signals = pd.Series(0, index=df.index)
            if preds > self.prediction_threshold:
                signals.iloc[-1] = 1  # Buy signal
            elif preds < (1 - self.prediction_threshold):
                signals.iloc[-1] = -1  # Sell signal
                
            return signals
        return pd.Series(0, index=df.index)