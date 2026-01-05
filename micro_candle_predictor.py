import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import asyncio
from collections import deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MicroCandlePredictor:
    def __init__(self, base_timeframe: str = '1m', micro_timeframe: float = 0.1):
        """
        Initialize the micro candle predictor.
        
        Args:
            base_timeframe: Base candle timeframe (e.g., '1m', '5m')
            micro_timeframe: Micro candle timeframe in seconds (e.g., 0.1 for 100ms)
        """
        self.base_timeframe = self._parse_timeframe(base_timeframe)
        self.micro_timeframe = micro_timeframe
        self.tick_buffer = deque(maxlen=1000)  # Store recent ticks
        self.micro_candles = deque(maxlen=100)  # Store recent micro-candles
        self.partial_candle = None
        self.last_prediction = None
        self.prediction_interval = 0.2  # Predict every 200ms
        self.last_prediction_time = 0
        
    def _parse_timeframe(self, tf: str) -> int:
        """Convert timeframe string to seconds."""
        unit = tf[-1].lower()
        value = int(tf[:-1])
        if unit == 's':
            return value
        elif unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    async def process_tick(self, price: float, volume: float, timestamp: Optional[float] = None):
        """Process a new tick and update micro-candles."""
        timestamp = timestamp or time.time()
        self.tick_buffer.append((timestamp, price, volume))
        
        # Update current micro-candle
        if self.partial_candle is None:
            self.partial_candle = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume,
                'start_time': timestamp,
                'tick_count': 1
            }
        else:
            self.partial_candle['high'] = max(self.partial_candle['high'], price)
            self.partial_candle['low'] = min(self.partial_candle['low'], price)
            self.partial_candle['close'] = price
            self.partial_candle['volume'] += volume
            self.partial_candle['tick_count'] += 1
            
        # Check if we need to finalize the current micro-candle
        current_time = timestamp
        elapsed = current_time - self.partial_candle['start_time']
        
        if elapsed >= self.micro_timeframe:
            self.partial_candle['end_time'] = current_time
            self.micro_candles.append(self.partial_candle)
            self.partial_candle = None

    def get_micro_candles(self, n: int = 10) -> List[Dict]:
        """Get the last n micro-candles."""
        return list(self.micro_candles)[-n:]

    def predict_micro_moves(self, model, features: np.ndarray) -> Dict:
        """Predict the next micro-move using the provided model."""
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            return self.last_prediction
            
        # Prepare features for the model
        prediction = model.predict(features)
        
        # Update last prediction time and cache
        self.last_prediction_time = current_time
        self.last_prediction = {
            'timestamp': current_time,
            'prediction': prediction,
            'confidence': float(np.max(prediction))
        }
        return self.last_prediction

    def get_candle_projection(self, current_candle: Dict, prediction: Dict) -> Dict:
        """Project the current candle based on micro-move predictions."""
        current_price = current_candle['close']
        predicted_move = prediction['prediction']
        confidence = prediction['confidence']
        
        # Calculate projected high/low/close based on prediction
        projected_high = max(current_candle['high'], current_price * (1 + predicted_move[0] * confidence))
        projected_low = min(current_candle['low'], current_price * (1 - predicted_move[1] * confidence))
        projected_close = current_price * (1 + (predicted_move[2] * 2 - 1) * confidence)
        
        return {
            'projected_high': projected_high,
            'projected_low': projected_low,
            'projected_close': projected_close,
            'confidence': confidence,
            'timestamp': time.time()
        }

    async def monitor_expiry(self, expiry_time: float, check_interval: float = 0.1):
        """Monitor the market as we approach expiry."""
        while time.time() < expiry_time:
            time_to_expiry = expiry_time - time.time()
            
            if time_to_expiry <= 10:  # Last 10 seconds
                # Increase prediction frequency
                self.prediction_interval = 0.05  # 50ms
            elif time_to_expiry <= 30:  # Last 30 seconds
                self.prediction_interval = 0.1  # 100ms
                
            await asyncio.sleep(check_interval)

    def calculate_optimal_entry(self, current_candle: Dict, prediction: Dict) -> Dict:
        """Calculate optimal entry point based on micro-structure."""
        spread = 0.0002  # Example spread
        current_price = current_candle['close']
        predicted_high = prediction.get('projected_high', current_price)
        predicted_low = prediction.get('projected_low', current_price)
        
        # Calculate risk-reward ratios
        buy_sl = predicted_low * 0.999  # 0.1% below predicted low
        buy_tp = predicted_high * 1.001  # 0.1% above predicted high
        sell_sl = predicted_high * 1.001
        sell_tp = predicted_low * 0.999
        
        buy_rr = (buy_tp - current_price) / (current_price - buy_sl) if current_price > buy_sl else 0
        sell_rr = (current_price - sell_tp) / (sell_sl - current_price) if current_price < sell_sl else 0
        
        return {
            'buy': {
                'entry': current_price + spread,
                'stop_loss': buy_sl,
                'take_profit': buy_tp,
                'risk_reward': buy_rr
            },
            'sell': {
                'entry': current_price - spread,
                'stop_loss': sell_sl,
                'take_profit': sell_tp,
                'risk_reward': sell_rr
            },
            'timestamp': time.time()
        }