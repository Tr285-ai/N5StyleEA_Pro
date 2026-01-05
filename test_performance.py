import pytest
import pandas as pd
import numpy as np
import time
import asyncio
from typing import Dict, Any
from pathlib import Path
import logging
import sys

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

from data_manager import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.mark.performance
class TestPerformance:
    """Performance test suite for the trading system."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment."""
        self.temp_dir = tmp_path / "test_data"
        self.temp_dir.mkdir()
        self.dm = DataManager(data_dir=self.temp_dir)
        
    def generate_large_dataframe(self, rows: int = 1_000_000) -> pd.DataFrame:
        """Generate a large DataFrame for testing."""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=rows, freq='s'),
            'open': np.random.normal(30000, 5000, rows).cumsum(),
            'high': np.random.normal(31000, 5000, rows).cumsum(),
            'low': np.random.normal(29000, 5000, rows).cumsum(),
            'close': np.random.normal(30000, 5000, rows).cumsum(),
            'volume': np.random.uniform(1000, 10000, rows)
        })

    @pytest.mark.benchmark(group="data_saving")
    def test_save_large_dataframe(self, benchmark):
        """Test saving a large DataFrame to disk."""
        df = self.generate_large_dataframe(1_000_000)
        
        def save_func():
            self.dm.save_data(df, "large_data", "parquet", compression='snappy')
            
        # Run benchmark
        benchmark(save_func)
        
        # Verify the file was created
        assert (self.temp_dir / "large_data.parquet").exists()

    @pytest.mark.benchmark(group="data_loading")
    def test_load_large_dataframe(self, benchmark):
        """Test loading a large DataFrame from disk."""
        # First save the data
        df = self.generate_large_dataframe(1_000_000)
        self.dm.save_data(df, "large_data_load", "parquet")
        
        # Benchmark loading
        def load_func():
            return self.dm.load_data("large_data_load.parquet")
            
        loaded_df = benchmark(load_func)
        assert loaded_df is not None
        assert len(loaded_df) == 1_000_000

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="concurrent_operations")
    async def test_concurrent_operations(self, benchmark):
        """Test concurrent read/write operations."""
        df = self.generate_large_dataframe(100_000)
        
        async def save_load_operations():
            tasks = []
            for i in range(10):
                tasks.append(asyncio.create_task(
                    self.async_save_load(f"data_{i}", df)
                ))
            await asyncio.gather(*tasks)
            
        # Run benchmark
        await benchmark(lambda: asyncio.run(save_load_operations()))
        
    async def async_save_load(self, name: str, df: pd.DataFrame):
        """Helper method for async save/load operations."""
        self.dm.save_data(df, name, "parquet")
        loaded = self.dm.load_data(f"{name}.parquet")
        assert loaded is not None
        assert len(loaded) == len(df)

    @pytest.mark.benchmark(group="memory_usage")
    def test_memory_usage(self, benchmark):
        """Test memory usage with large datasets."""
        df = self.generate_large_dataframe(5_000_000)
        
        def process_data():
            # Perform some memory-intensive operations
            result = df.rolling(window=100).mean()
            result['sma'] = df['close'].rolling(window=50).mean()
            result['rsi'] = self.calculate_rsi(df['close'])
            return result
            
        # Run benchmark
        result = benchmark(process_data)
        assert result is not None
        
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @pytest.mark.benchmark(group="data_cleaning")
    def test_data_cleaning_performance(self, benchmark):
        """Test performance of data cleaning operations."""
        df = self.generate_large_dataframe(1_000_000)
        
        def clean_data():
            # Add some missing values
            df_clean = df.copy()
            mask = np.random.random(len(df)) < 0.1  # 10% missing values
            df_clean.loc[mask, 'volume'] = np.nan
            
            # Fill missing values
            df_clean['volume'] = df_clean['volume'].fillna(method='ffill')
            
            # Remove outliers
            q_low = df_clean['close'].quantile(0.01)
            q_high = df_clean['close'].quantile(0.99)
            df_clean = df_clean[
                (df_clean['close'] >= q_low) & 
                (df_clean['close'] <= q_high)
            ]
            return df_clean
            
        # Run benchmark
        cleaned_df = benchmark(clean_data)
        assert cleaned_df is not None
        assert len(cleaned_df) <= len(df)

if __name__ == "__main__":
    # Run tests with: python -m pytest test_performance.py -v --benchmark-only
    pytest.main(["-v", "--benchmark-only"])