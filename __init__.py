"""
N5StyleEA Pro v15.3 - Main Package

This package contains the core functionality for the N5StyleEA Pro trading system.
"""

__version__ = "15.3.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import core modules
try:
    from . import core  # type: ignore
    from . import strategies  # type: ignore
    from . import utils  # type: ignore
    from . import api  # type: ignore
except Exception:
    try:
        import core  # type: ignore
        import strategies  # type: ignore
        import utils  # type: ignore
        import api  # type: ignore
    except Exception:
        core = None  # type: ignore
        strategies = None  # type: ignore
        utils = None  # type: ignore
        api = None  # type: ignore

# Initialize logging
import logging

def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('n5styleea.log')
        ]
    )

# Set up default logging
setup_logging()

# Import main components
try:
    from .app import app  # type: ignore
    from .core.trading_engine import TradingEngine  # type: ignore
    from .core.api.trade_api import TradeAPI  # type: ignore
except Exception:
    try:
        from app import app  # type: ignore
        from core.trading_engine import TradingEngine  # type: ignore
        from core.api.trade_api import TradeAPI  # type: ignore
    except Exception:
        app = None  # type: ignore
        TradingEngine = None  # type: ignore
        TradeAPI = None  # type: ignore

__all__ = [
    'app',
    'TradingEngine',
    'TradeAPI',
    'setup_logging'
]