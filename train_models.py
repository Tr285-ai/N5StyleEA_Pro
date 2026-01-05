# train_models.py
import asyncio
import logging
from datetime import datetime, timedelta
from trading_system import TradingSystem
from config import load_config

async def train_models_loop():
    """Periodically retrain ML models."""
    config = load_config('config/config.yaml')
    ml_config = load_config('config/ml_config.yaml')
    config['ml'] = ml_config['ml']
    
    trading_system = TradingSystem(config)
    await trading_system.initialize()
    
    while True:
        try:
            # Load new data
            historical_data = load_historical_data()
            
            # Retrain models
            await trading_system.ml_integration.train_models(historical_data)
            
            # Wait for next training cycle
            await asyncio.sleep(ml_config['ml']['ensemble']['retrain_interval'])
            
        except Exception as e:
            logging.error(f"Error in training loop: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(train_models_loop())