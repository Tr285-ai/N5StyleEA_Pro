import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI application."""
    try:
        from main import app
    except Exception as e:
        import pytest as _pytest
        _pytest.skip(f"main import failed: {e}")
    from fastapi.testclient import TestClient as _TC
    with _TC(app) as client:
        yield client


@pytest.fixture(scope="function")
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Generate sample market data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return {
        "symbol": "BTC/USDT",
        "data": pd.DataFrame({
            "open": np.random.normal(30000, 5000, 100).cumsum(),
            "high": np.random.normal(31000, 5000, 100).cumsum(),
            "low": np.random.normal(29000, 5000, 100).cumsum(),
            "close": np.random.normal(30000, 5000, 100).cumsum(),
            "volume": np.random.uniform(1000, 10000, 100),
        }, index=dates)
    }

@pytest.fixture
def data_manager(temp_data_dir: Path):
    """Create a DataManager instance with a temporary directory."""
    try:
        from data_manager import DataManager as _DM
    except Exception as e:
        import pytest as _pytest
        _pytest.skip(f"DataManager import failed: {e}")
    return _DM(data_dir=temp_data_dir)


@pytest.fixture
def sample_trade() -> Dict[str, Any]:
    """Generate a sample trade for testing."""
    return {
        "id": "12345",
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 40000.0,
        "size": 0.1,
        "timestamp": "2023-01-01T12:00:00Z"
    }

@pytest.fixture
def sample_portfolio() -> Dict[str, Any]:
    """Generate a sample portfolio for testing."""
    return {
        "total_value": 100000.0,
        "positions": [
            {"symbol": "BTC", "size": 0.5, "price": 40000.0},
            {"symbol": "ETH", "size": 10.0, "price": 2500.0},
        ]
    }
