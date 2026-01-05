# config/risk_config.py
RISK_CONFIG = {
    # Base risk parameters
    'account_balance': 10000.0,  # Initial account balance
    'risk_per_trade': 0.01,      # 1% risk per trade
    'max_drawdown_pct': 5.0,     # 5% max daily drawdown
    'max_trades_per_day': 20,    # Maximum trades per day
    
    # Position sizing
    'max_position_size_pct': 10.0,  # Maximum position size as % of account
    'max_leverage': 5.0,           # Maximum allowed leverage
    
    # Regime-based adjustments
    'regime_factors': {
        'trending': 1.2,
        'ranging': 0.8,
        'volatile': 0.5,
        'unknown': 0.7
    },
    
    # Liquidity thresholds
    'liquidity': {
        'high': {
            'min_volume': 1000,
            'max_spread': 0.0005
        },
        'medium': {
            'min_volume': 500,
            'max_spread': 0.001
        },
        'low': {
            'min_volume': 100,
            'max_spread': 0.002
        }
    }
}