# ai_assistant_integration.py
import openai
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class AIMarketAnalysis:
    sentiment: str
    confidence: float
    reasoning: str
    suggested_actions: list

class AIAssistant:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize the AI assistant with API key and model."""
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> AIMarketAnalysis:
        """
        Analyze market conditions using AI.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            AIMarketAnalysis object with analysis results
        """
        try:
            prompt = self._create_analysis_prompt(market_data)
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            analysis = self._parse_ai_response(response)
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            raise
    
    def _create_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create a prompt for the AI based on market data."""
        return f"""
        Analyze the following market data and provide trading insights:
        {json.dumps(market_data, indent=2)}
        
        Please provide:
        1. Market sentiment (bullish/bearish/neutral)
        2. Confidence level (0-1)
        3. Brief reasoning
        4. Suggested trading actions
        """
    
    def _parse_ai_response(self, response: Dict[str, Any]) -> AIMarketAnalysis:
        """Parse the AI response into a structured format."""
        content = response.choices[0].message.content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Simple parsing logic - adjust based on your needs
        sentiment = "neutral"
        confidence = 0.5
        reasoning = []
        actions = []
        
        for line in lines:
            line_lower = line.lower()
            if "sentiment:" in line_lower:
                sentiment = line.split(":")[1].strip().lower()
            elif "confidence:" in line_lower:
                try:
                    confidence = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "reasoning:" in line_lower:
                reasoning = [line.split(":")[1].strip()]
            elif "action:" in line_lower:
                actions.append(line.split(":")[1].strip())
        
        return AIMarketAnalysis(
            sentiment=sentiment,
            confidence=confidence,
            reasoning=" ".join(reasoning) if reasoning else "No reasoning provided",
            suggested_actions=actions
        )
    
    async def get_trade_recommendation(self, symbol: str, price: float) -> Dict[str, Any]:
        """Get a trade recommendation for a given symbol and price."""
        market_data = {
            "symbol": symbol,
            "price": price,
            "timestamp": str(datetime.utcnow())
        }
        
        analysis = await self.analyze_market(market_data)
        
        return {
            "symbol": symbol,
            "action": "buy" if analysis.sentiment == "bullish" else "sell",
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "timestamp": str(datetime.utcnow())
        }