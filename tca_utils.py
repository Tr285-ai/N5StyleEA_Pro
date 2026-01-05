from typing import Optional

def compute_slippage_bps(arrival_price: Optional[float], fill_price: Optional[float], side: str) -> Optional[float]:
    """
    Compute slippage in basis points (bps).
    Positive = worse than arrival (cost), Negative = improvement.
    For buys: (fill - arrival) / arrival * 10_000
    For sells: (arrival - fill) / arrival * 10_000
    """
    try:
        if arrival_price is None or fill_price is None or arrival_price <= 0:
            return None
        s = side.lower().strip()
        if s == 'buy':
            return (float(fill_price) - float(arrival_price)) / float(arrival_price) * 10_000.0
        elif s == 'sell':
            return (float(arrival_price) - float(fill_price)) / float(arrival_price) * 10_000.0
        else:
            return None
    except Exception:
        return None
