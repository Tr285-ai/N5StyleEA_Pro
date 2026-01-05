import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from economic_calendar import EconomicCalendar
from model_registry import ModelEnsemble, ModelConfig
from time_filter import TimeFilter, MarketSession
from logging_json import log_json

try:
    from notifications import EmailNotifier
except Exception:
    EmailNotifier = None

logger = logging.getLogger(__name__)


class TradingStrategy:
    def __init__(
        self,
        model_path: str,
        input_shape: Tuple[int, int] = (30, 5),
        email_config: Optional[Dict[str, Any]] = None,
        econ_calendar_api_key: Optional[str] = None,
    ):
        self.model_path = model_path
        self.input_shape = input_shape

        self.time_filter = TimeFilter()
        self.econ_calendar = EconomicCalendar(api_key=econ_calendar_api_key) if econ_calendar_api_key else None

        self.email_notifier = None
        if email_config and EmailNotifier is not None:
            try:
                self.email_notifier = EmailNotifier(
                    smtp_server=email_config.get('smtp_server'),
                    smtp_port=int(email_config.get('smtp_port') or 0),
                    username=email_config.get('username') or email_config.get('email'),
                    password=email_config.get('password'),
                    sender_email=email_config.get('sender_email') or email_config.get('email'),
                    admin_email=email_config.get('admin_email') or email_config.get('email'),
                )
            except Exception as e:
                logger.error(f"Failed to initialize EmailNotifier: {e}")

        self._initialize_model()

    def _initialize_model(self) -> None:
        model_configs = [
            ModelConfig(
                model_type='lstm',
                params={
                    'input_shape': self.input_shape,
                    'units': 64,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                },
                weight=0.6,
            ),
            ModelConfig(
                model_type='cnn',
                params={
                    'input_shape': self.input_shape,
                    'filters': 64,
                    'kernel_size': 3,
                    'learning_rate': 0.001,
                },
                weight=0.4,
            ),
        ]

        try:
            if os.path.exists(f"{self.model_path}_weights.pkl"):
                self.model_ensemble = ModelEnsemble.load(self.model_path, model_configs)
                logger.info(f"Loaded model ensemble from {self.model_path}")
            else:
                self.model_ensemble = ModelEnsemble(model_configs)
                logger.warning(f"No saved model found at {self.model_path}; using untrained ensemble")
        except Exception as e:
            logger.error(f"Error initializing model ensemble: {e}")
            raise

    async def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        try:
            if market_data is None or market_data.empty:
                return self._hold("no_market_data")

            session = self.time_filter.get_market_session(datetime.now(timezone.utc))
            if session == MarketSession.CLOSED:
                return self._hold("market_closed")

            if self.econ_calendar and self.econ_calendar.is_high_impact_period(symbol.replace("/", "")):
                return self._hold("high_impact_news")

            X = self._prepare_features(market_data)
            pred = self.model_ensemble.predict(X)
            score = float(np.asarray(pred).reshape(-1)[0])

            price = float(market_data['close'].iloc[-1])
            ts = datetime.now(timezone.utc).isoformat()

            if score >= 0.6:
                out = {
                    'symbol': symbol,
                    'signal': 'BUY',
                    'confidence': score,
                    'timestamp': ts,
                    'price': price,
                    'session': session.value,
                }
                try:
                    log_json('signal_generated', symbol=symbol, signal='SELL', confidence=1.0 - score, price=price, session=session.value)
                except Exception:
                    pass
                return out
            if score <= 0.4:
                out = {
                    'symbol': symbol,
                    'signal': 'SELL',
                    'confidence': 1.0 - score,
                    'timestamp': ts,
                    'price': price,
                    'session': session.value,
                }
                try:
                    log_json('signal_generated', symbol=symbol, signal='SELL', confidence=1.0 - score, price=price, session=session.value)
                except Exception:
                    pass
                return out
            out = self._hold("no_edge", session=session.value, price=price)
            try:
                log_json('signal_generated', symbol=symbol, signal=out.get('signal','HOLD'), confidence=out.get('confidence',0.5), price=price, session=session.value, reason=out.get('reason'))
            except Exception:
                pass
            return out

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            try:
                log_json('signal_error', symbol=symbol, error=str(e))
            except Exception:
                pass
            return self._hold("signal_error")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        for c in base_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        window = df[base_cols].tail(self.input_shape[0]).copy()
        if len(window) < self.input_shape[0]:
            pad = pd.DataFrame(
                np.zeros((self.input_shape[0] - len(window), len(base_cols))),
                columns=base_cols,
            )
            window = pd.concat([pad, window], ignore_index=True)

        values = window.values.astype(np.float32)
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True) + 1e-8
        values = (values - mean) / std

        return values.reshape(1, *self.input_shape)

    def _hold(self, reason: str, **extra: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'signal': 'HOLD',
            'confidence': 0.5,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'reason': reason,
        }
        out.update(extra)
        return out