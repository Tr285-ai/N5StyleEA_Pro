# economic_calendar.py
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import pytz

logger = logging.getLogger(__name__)

@dataclass
class EconomicEvent:
    title: str
    country: str
    impact: str  # 'High', 'Medium', 'Low'
    timestamp: datetime
    previous: Optional[float] = None
    forecast: Optional[float] = None
    actual: Optional[float] = None

class EconomicCalendar:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.timezone = pytz.UTC
        
    def get_upcoming_events(
        self,
        days_ahead: int = 2,
        min_impact: str = 'Medium',
        countries: List[str] = None
    ) -> List[EconomicEvent]:
        """
        Get upcoming economic events.
        
        Args:
            days_ahead: Number of days to look ahead
            min_impact: Minimum impact level to include ('High', 'Medium', 'Low')
            countries: List of country codes to filter by (e.g., ['US', 'EU', 'GB'])
        """
        # This is a placeholder - implement with your preferred economic calendar API
        # Example: Forex Factory, Investing.com, or a paid service like Alpha Vantage
        
        # Example implementation with Forex Factory (would need API key)
        if not self.api_key:
            logger.warning("No API key provided for economic calendar")
            return []
            
        end_date = datetime.utcnow() + timedelta(days=days_ahead)
        
        try:
            # This is a placeholder - replace with actual API call
            # response = requests.get(
            #     f"https://api.forexfactory.com/v1/calendar",
            #     params={
            #         'api_key': self.api_key,
            #         'end': end_date.strftime('%Y-%m-%d')
            #     }
            # )
            # events = self._parse_events(response.json())
            
            # Filter by impact and countries
            events = []  # Replace with actual events from API
            if min_impact == 'High':
                events = [e for e in events if e.impact == 'High']
            elif min_impact == 'Medium':
                events = [e for e in events if e.impact in ['High', 'Medium']]
                
            if countries:
                events = [e for e in events if e.country in countries]
                
            return events
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return []
    
    def is_high_impact_period(self, symbol: str, minutes_before: int = 30, minutes_after: int = 60) -> bool:
        """Check if we're in a high-impact economic event window for the given symbol."""
        if not self.api_key:
            return False
            
        # Map symbols to countries/regions
        symbol_to_country = {
            'EURUSD': ['EU', 'US'],
            'GBPUSD': ['GB', 'US'],
            'USDJPY': ['US', 'JP'],
            # Add more mappings as needed
        }
        
        countries = symbol_to_country.get(symbol, [])
        if not countries:
            return False
            
        now = datetime.utcnow()
        events = self.get_upcoming_events(
            days_ahead=1,
            min_impact='High',
            countries=countries
        )
        
        for event in events:
            event_start = event.timestamp - timedelta(minutes=minutes_before)
            event_end = event.timestamp + timedelta(minutes=minutes_after)
            
            if event_start <= now <= event_end:
                logger.info(f"High impact event: {event.title} at {event.timestamp}")
                return True
                
        return False