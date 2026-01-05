import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

_logger = logging.getLogger(__name__)

def log_json(event: str, level: int = logging.INFO, **fields: Any) -> None:
    try:
        record: Dict[str, Any] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            **fields,
        }
        logging.log(level, json.dumps(record, ensure_ascii=False))
    except Exception:
        # Never break caller code due to logging
        try:
            _logger.log(level, f"{event} {fields}")
        except Exception:
            pass
