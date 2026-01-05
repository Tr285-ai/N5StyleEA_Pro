import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Suppress some noisy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ParameterRange:
    """Defines a range for a parameter to be optimized."""
    name: str
    type: str  # 'int', 'float', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    categories: Optional[List[Any]] = None
    
    def suggest(self, trial: optuna.Trial, name: Optional[str] = None) -> Any:
        """Suggest a value for this parameter using the trial object."""
        name = name or self.name
        if self.type == 'int':
            return trial.suggest_int(name, int(self.low), int(self.high), step=int(self.step or 1))
        elif self.type == 'float':
            return trial.suggest_float(name, self.low, self.high, step=self.step)
        elif self.type == 'categorical' and self.categories is not None:
            return trial.suggest_categorical(name, self.categories)
        else:
            raise ValueError(f"Invalid parameter type or configuration: {self}")

@dataclass
class AutoTrainConfig:
    """Configuration for auto-training a trading strategy."""
    # Strategy and data configuration
    strategy_class: Any
    data_provider: Any
    symbol: str
    timeframe: str
    params_config: Dict[str, Dict[str, Any]]
    
    # Date ranges
    train_start: Union[str, datetime]
    train_end: Union[str, datetime]
    val_start: Optional[Union[str, datetime]] = None
    val_end: Optional[Union[str, datetime]] = None
    test_start: Optional[Union[str, datetime]] = None
    test_end: Optional[Union[str, datetime]] = None
    
    # Optimization settings
    n_trials: int = 100
    n_jobs: int = -1  # -1 for all available cores
    timeout: Optional[int] = 3600  # seconds
    metric: str = 'sharpe_ratio'  # Metric to optimize
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    
    # Early stopping
    early_stopping_rounds: Optional[int] = 10
    min_trials: int = 20
    
    # Output settings
    output_dir: str = "autotrain_results"
    save_best_params: bool = True
    save_all_trials: bool = True
    save_plots: bool = True
    
    def __post_init__(self):
        """Validate and convert configuration values."""
        # Convert string dates to datetime
        if isinstance(self.train_start, str):
            self.train_start = pd.to_datetime(self.train_start)
        if isinstance(self.train_end, str):
            self.train_end = pd.to_datetime(self.train_end)
        if isinstance(self.val_start, str):
            self.val_start = pd.to_datetime(self.val_start)
        if isinstance(self.val_end, str):
            self.val_end = pd.to_datetime(self.val_end)
        if isinstance(self.test_start, str):
            self.test_start = pd.to_datetime(self.test_start)
        if isinstance(self.test_end, str):
            self.test_end = pd.to_datetime(self.test_end)
            
        # Set default validation and test periods if not provided
        if self.val_start is None:
            self.val_start = self.train_end
        if self.val_end is None and self.test_start is not None:
            self.val_end = self.test_start
        elif self.val_end is None:
            self.val_end = self.train_end + (self.train_end - self.train_start) / 2
            
        if self.test_start is None and self.val_end is not None:
            self.test_start = self.val_end
        if self.test_end is None and self.test_start is not None:
            self.test_end = self.test_start + (self.val_end - self.val_start)
            
        # Validate date ranges
        if self.train_start >= self.train_end:
            raise ValueError("train_start must be before train_end")
        if self.val_start is not None and self.val_end is not None and self.val_start >= self.val_end:
            raise ValueError("val_start must be before val_end")
        if self.test_start is not None and self.test_end is not None and self.test_start >= self.test_end:
            raise ValueError("test_start must be before test_end")
            
        # Validate metric and direction
        if self.direction not in ['maximize', 'minimize']:
            raise ValueError("direction must be either 'maximize' or 'minimize'")
            
        # Set number of jobs
        if self.n_jobs == -1:
            import multiprocessing
            self.n_jobs = multiprocessing.cpu_count()
            
        # Create output directory
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class AutoTrainer:
    """Automated hyperparameter optimization for trading strategies."""
    
    def __init__(self, config: AutoTrainConfig):
        """Initialize the auto-trainer with configuration."""
        self.config = config
        self.study = None
        self.best_params = None
        self.best_value = None
        self.trials_df = None
        
    def _create_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Create parameter ranges from the configuration."""
        param_ranges = {}
        
        for param_name, param_config in self.config.params_config.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                param_ranges[param_name] = ParameterRange(
                    name=param_name,
                    type='int',
                    low=param_config.get('min'),
                    high=param_config.get('max'),
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                param_ranges[param_name] = ParameterRange(
                    name=param_name,
                    type='float',
                    low=param_config.get('min'),
                    high=param_config.get('max'),
                    step=param_config.get('step')
                )
            elif param_type == 'categorical':
                param_ranges[param_name] = ParameterRange(
                    name=param_name,
                    type='categorical',
                    categories=param_config.get('choices', [])
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
                
        return param_ranges
        
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        try:
            # Suggest parameter values
            params = {}
            param_ranges = self._create_parameter_ranges()
            
            for param_name, param_range in param_ranges.items():
                params[param_name] = param_range.suggest(trial)
                
            # Create and initialize strategy
            strategy = self.config.strategy_class(**params)
            
            # Run backtest on training data
            train_metrics = self._evaluate_strategy(
                strategy,
                self.config.train_start,
                self.config.train_end
            )
            
            if train_metrics is None:
                raise ValueError("Training evaluation failed")
                
            # Run backtest on validation data
            val_metrics = self._evaluate_strategy(
                strategy,
                self.config.val_start,
                self.config.val_end
            )
            
            if val_metrics is None:
                raise ValueError("Validation evaluation failed")
                
            # Get the metric to optimize
            metric_value = val_metrics.get(self.config.metric)
            
            if metric_value is None:
                raise ValueError(f"Metric '{self.config.metric}' not found in validation results")
                
            # Store additional information
            trial.set_user_attr('train_metrics', train_metrics)
            trial.set_user_attr('val_metrics', val_metrics)
            trial.set_user_attr('params', params)
            
            return metric_value if self.config.direction == 'maximize' else -metric_value
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            # Return a very bad score if the trial fails
            return float('-inf') if self.config.direction == 'maximize' else float('inf')
            
    def _evaluate_strategy(
        self,
        strategy: Any,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, float]]:
        """Evaluate a strategy on the given date range."""
        try:
            # This is a simplified example - you would typically use your backtesting engine here
            # Replace this with your actual backtesting code
            logger.info(f"Evaluating strategy from {start_date} to {end_date}")
            
            # Simulate some metrics (replace with actual backtest)
            metrics = {
                'sharpe_ratio': random.uniform(-1, 3),
                'sortino_ratio': random.uniform(-1, 3),
                'total_return': random.uniform(-50, 200),
                'max_drawdown': random.uniform(0, 50),
                'win_rate': random.uniform(30, 70),
                'profit_factor': random.uniform(0.5, 3.0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            return None
            
    async def optimize(self) -> Dict[str, Any]:
        """Run the optimization process."""
        try:
            logger.info("Starting hyperparameter optimization")
            
            # Create study
            sampler = TPESampler(n_startup_trials=10, multivariate=True)
            pruner = SuccessiveHalvingPruner()
            
            self.study = optuna.create_study(
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner
            )
            
            # Run optimization
            self.study.optimize(
                self._objective,
                n_trials=self.config.n_trials,
                n_jobs=self.config.n_jobs,
                timeout=self.config.timeout,
                show_progress_bar=True
            )
            
            # Get best parameters
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            
            # Save results
            await self._save_results()
            
            logger.info("Optimization completed successfully")
            return self.best_params
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
            
    async def _save_results(self) -> None:
        """Save optimization results."""
        if self.study is None:
            return
            
        # Save best parameters
        if self.config.save_best_params and self.best_params:
            best_params_file = self.config.output_dir / "best_params.json"
            with open(best_params_file, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            logger.info(f"Saved best parameters to {best_params_file}")
            
        # Save all trials
        if self.config.save_all_trials:
            trials_file = self.config.output_dir / "trials.csv"
            self.trials_df = self.study.trials_dataframe()
            self.trials_df.to_csv(trials_file, index=False)
            logger.info(f"Saved trial results to {trials_file}")
            
        # Save study
        study_file = self.config.output_dir / "study.pkl"
        import joblib
        joblib.dump(self.study, study_file)
        logger.info(f"Saved study to {study_file}")
        
        # Save visualizations
        if self.config.save_plots:
            try:
                import optuna.visualization as vis
                
                # Optimization history
                fig = vis.plot_optimization_history(self.study)
                fig.write_image(str(self.config.output_dir / "optimization_history.png"))
                
                # Parameter importance
                try:
                    fig = vis.plot_param_importances(self.study)
                    fig.write_image(str(self.config.output_dir / "param_importances.png"))
                except:
                    logger.warning("Could not generate parameter importance plot")
                    
                # Parallel coordinate plot
                try:
                    fig = vis.plot_parallel_coordinate(self.study)
                    fig.write_image(str(self.config.output_dir / "parallel_coordinate.png"))
                except:
                    logger.warning("Could not generate parallel coordinate plot")
                    
            except Exception as e:
                logger.warning(f"Could not generate visualizations: {e}")
                
    def test_best_params(self) -> Dict[str, Any]:
        """Test the best parameters on the test set."""
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
            
        if self.config.test_start is None or self.config.test_end is None:
            logger.warning("No test period defined. Skipping test.")
            return {}
            
        try:
            logger.info("Testing best parameters on test set")
            
            # Create strategy with best parameters
            strategy = self.config.strategy_class(**self.best_params)
            
            # Run backtest on test data
            test_metrics = self._evaluate_strategy(
                strategy,
                self.config.test_start,
                self.config.test_end
            )
            
            if test_metrics is not None:
                # Save test metrics
                test_metrics_file = self.config.output_dir / "test_metrics.json"
                with open(test_metrics_file, 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Saved test metrics to {test_metrics_file}")
                
            return test_metrics or {}
            
        except Exception as e:
            logger.error(f"Error testing best parameters: {e}")
            raise

async def main():
    """Example usage of the AutoTrainer."""
    # Example configuration
    config = AutoTrainConfig(
        strategy_class=MovingAverageCrossover,  # Replace with your strategy class
        data_provider=None,  # Replace with your data provider
        symbol="BTC/USDT",
        timeframe="1d",
        params_config={
            'fast_window': {'type': 'int', 'min': 5, 'max': 50},
            'slow_window': {'type': 'int', 'min': 20, 'max': 200},
            'trailing_stop': {'type': 'float', 'min': 0.5, 'max': 5.0, 'step': 0.1},
            'rsi_period': {'type': 'int', 'min': 7, 'max': 21},
            'rsi_overbought': {'type': 'int', 'min': 60, 'max': 90},
            'rsi_oversold': {'type': 'int', 'min': 10, 'max': 40}
        },
        train_start="2020-01-01",
        train_end="2021-12-31",
        val_start="2022-01-01",
        val_end="2022-06-30",
        test_start="2022-07-01",
        test_end="2022-12-31",
        n_trials=100,
        n_jobs=-1,
        output_dir="autotrain_results"
    )
    
    # Create and run auto-trainer
    trainer = AutoTrainer(config)
    best_params = await trainer.optimize()
    
    # Test best parameters
    test_metrics = trainer.test_best_params()
    
    print("\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    asyncio.run(main())

    def _evaluate_strategy(
        self,
        strategy: Any,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, float]]:
        """Evaluate a strategy on the given date range."""
        try:
            # This is a simplified example - you would typically use your backtesting engine here
            # Replace this with your actual backtesting code
            logger.info(f"Evaluating strategy from {start_date} to {end_date}")
            
            # Simulate some metrics (replace with actual backtest)
            metrics = {
                'sharpe_ratio': random.uniform(-1, 3),
                'sortino_ratio': random.uniform(-1, 3),
                'total_return': random.uniform(-50, 200),
                'max_drawdown': random.uniform(0, 50),
                'win_rate': random.uniform(30, 70),
                'profit_factor': random.uniform(0.5, 3.0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            return None    