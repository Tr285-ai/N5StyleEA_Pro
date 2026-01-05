# config/strategy_config.py
STRATEGY_CONFIG = {
    # Momentum Strategy
    'momentum': {
        'enabled': True,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'risk_per_trade': 1.0
    },
    
    # Moving Average Crossover
    'ma_crossover': {
        'enabled': True,
        'fast_ma': 10,
        'slow_ma': 30,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'risk_per_trade': 1.0,
        'trailing_stop': True,
        'trailing_stop_atr_mult': 1.5
    },
    
    # Pairs Trading
    'pairs_trading': {
        'enabled': True,
        'lookback': 20,
        'entry_z': 2.0,
        'exit_z': 0.5,
        'max_holding_period': 10,
        'adf_pvalue': 0.05,
        'risk_per_trade': 0.5  # Lower risk for pairs trading
    },
    
    # Volatility Breakout
    'volatility_breakout': {
        'enabled': True,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'volatility_threshold': 0.01
    }
}