# ml_integration.py
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .ai_assistant_integration import AIAssistant
from .evolving_trader_combined import EvolvingTrader, EvolutionConfig
from .micro_predictor_v15_2 import MicroPredictor
from .train_micro_models import train_micro_models
from .train_micro_ensemble import MicroEnsembleTrainer

logger = logging.getLogger('ml_integration')

class MLIntegration:
    """Core class for AI/ML integration in the trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ai_assistant = None
        self.evolving_trader = None
        self.micro_predictor = None
        self.ensemble_trainer = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all ML components."""
        if self.initialized:
            return
            
        logger.info("Initializing ML components...")
        
        # Initialize AI Assistant
        self.ai_assistant = AIAssistant(
            api_key=self.config.get('openai_api_key'),
            model=self.config.get('ai_model', 'gpt-4')
        )
        
        # Initialize Evolving Trader
        input_size = self.config.get('evolving_trader', {}).get('input_size', 10)
        self.evolving_trader = EvolvingTrader(
            input_size=input_size,
            config=EvolutionConfig()
        )
        
        # Initialize Micro Predictor
        self.micro_predictor = MicroPredictor(
            model_dir=Path('models/micro_models'),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize Ensemble Trainer
        self.ensemble_trainer = MicroEnsembleTrainer(
            model_dir=Path('models/ensembles'),
            n_models=5
        )
        
        self.initialized = True
        logger.info("ML components initialized")
        
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-powered market analysis."""
        if not self.initialized:
            await self.initialize()
            
        # Get AI analysis
        ai_analysis = await self.ai_assistant.analyze_market_conditions(market_data)
        
        # Get predictions from evolving trader
        state = self._prepare_state(market_data)
        trader_prediction = self.evolving_trader.predict(state)
        
        # Get micro model predictions
        micro_prediction = await self.micro_predictor.predict(market_data)
        
        return {
            'ai_analysis': ai_analysis,
            'trader_prediction': trader_prediction,
            'micro_prediction': micro_prediction
        }
        
    async def train_models(self, data: pd.DataFrame):
        """Train all ML models."""
        if not self.initialized:
            await self.initialize()
            
        logger.info("Starting model training...")
        
        # Train evolving trader
        X = data.drop(columns=['target']).values
        y = data['target'].values
        self.evolving_trader.train(X, y)
        
        # Train micro models
        await train_micro_models(
            data,
            model_dir='models/micro_models',
            n_models=5
        )
        
        # Train ensemble
        await self.ensemble_trainer.train(data)
        
        logger.info("Model training completed")
        
    def _prepare_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare market data for model input."""
        # Convert market data to numpy array
        features = [
            market_data['open'],
            market_data['high'],
            market_data['low'],
            market_data['close'],
            market_data['volume'],
            market_data.get('rsi', 50),
            market_data.get('macd', 0),
            market_data.get('bollinger_upper', 0),
            market_data.get('bollinger_middle', 0),
            market_data.get('bollinger_lower', 0)
        ]
        return np.array(features, dtype=np.float32)