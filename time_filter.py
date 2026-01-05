from datetime import time, datetime, timedelta
from enum import Enum
import pytz
from typing import Dict, List, Optional
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)

class MarketSession(Enum):
    ASIA = "Asia"
    LONDON = "London"
    NEWYORK = "New York"
    SYDNEY = "Sydney"
    OVERLAP = "Overlap"
    CLOSED = "Closed"

class TimeFilter:
    def __init__(self, data_dir: str = "data"):
        self.timezone = pytz.timezone('UTC')
        self.sessions = {
            MarketSession.ASIA: {'start': time(0, 0), 'end': time(8, 0)},
            MarketSession.LONDON: {'start': time(8, 0), 'end': time(16, 0)},
            MarketSession.NEWYORK: {'start': time(13, 0), 'end': time(21, 0)},
            MarketSession.SYDNEY: {'start': time(22, 0), 'end': time(6, 0)},
        }
        self.overlap_windows = [
            (time(8, 0), time(9, 0)),    # London-Asia overlap
            (time(13, 0), time(16, 0)),   # London-NY overlap
            (time(21, 0), time(22, 0)),   # NY close - Sydney open
        ]
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.performance_data = self._load_performance_data()

    def _load_performance_data(self) -> pd.DataFrame:
        """Load historical performance data"""
        filepath = os.path.join(self.data_dir, 'performance.csv')
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, parse_dates=['timestamp'])
                return df
            except Exception as e:
                logger.error(f"Error loading performance data: {e}")
        return pd.DataFrame(columns=['timestamp', 'session', 'profit', 'trades'])

    def _save_performance_data(self) -> None:
        """Save performance data to disk"""
        try:
            filepath = os.path.join(self.data_dir, 'performance.csv')
            self.performance_data.to_csv(filepath, index=False)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")

    def get_market_session(self, dt: datetime) -> MarketSession:
        """Determine the current market session"""
        if not dt.tzinfo:
            dt = self.timezone.localize(dt)
        else:
            dt = dt.astimezone(self.timezone)
            
        current_time = dt.time()
        
        # Check for session overlaps first
        for start, end in self.overlap_windows:
            if self._time_in_range(start, end, current_time):
                return MarketSession.OVERLAP
                
        # Check regular sessions
        for session, times in self.sessions.items():
            if self._time_in_range(times['start'], times['end'], current_time):
                return session
                
        return MarketSession.CLOSED

    def _time_in_range(self, start: time, end: time, current: time) -> bool:
        """Check if current time is within range, handling overnight ranges"""
        if start <= end:
            return start <= current < end
        else:  # Overnight range (e.g., 22:00-06:00)
            return current >= start or current < end

    def update_performance(self, timestamp: datetime, profit: float) -> None:
        """Update performance metrics"""
        session = self.get_market_session(timestamp)
        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'session': session.value,
            'profit': profit,
            'trades': 1
        }])
        self.performance_data = pd.concat([self.performance_data, new_row], ignore_index=True)
        self._save_performance_data()

    def get_session_performance(self, days: int = 30) -> Dict[str, float]:
        """Get performance metrics by session for the last N days"""
        if self.performance_data.empty:
            return {session.value: 0.0 for session in MarketSession}
            
        cutoff = datetime.now(self.timezone) - timedelta(days=days)
        recent = self.performance_data[self.performance_data['timestamp'] >= cutoff]
        
        if recent.empty:
            return {session.value: 0.0 for session in MarketSession}
            
        return recent.groupby('session')['profit'].sum().to_dict()

    def plot_performance(self, days: int = 30) -> None:
        """Generate performance visualization"""
        if self.performance_data.empty:
            logger.warning("No performance data available")
            return
            
        plt.figure(figsize=(12, 6))
        recent = self.performance_data[
            self.performance_data['timestamp'] >= 
            (datetime.now(self.timezone) - timedelta(days=days))
        ]
        
        if not recent.empty:
            # Plot daily profit
            daily = recent.set_index('timestamp').resample('D')['profit'].sum()
            daily.plot(kind='line', title='Daily Profit/Loss')
            plt.axhline(0, color='red', linestyle='--')
            plt.tight_layout()
            
            # Save the plot
            os.makedirs('reports', exist_ok=True)
            plt.savefig('reports/performance.png')
            plt.close()