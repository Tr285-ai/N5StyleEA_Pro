import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataManager with a data directory.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save_data(
        self,
        data: Union[pd.DataFrame, Dict, List],
        filename: str,
        file_format: str = 'parquet',
        **kwargs
    ) -> bool:
        """
        Save data to disk in the specified format.
        
        Args:
            data: Data to save (DataFrame, dict, or list)
            filename: Name of the file (without extension)
            file_format: Format to save as ('parquet', 'csv', 'json')
            **kwargs: Additional arguments to pass to the save function
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            filepath = self.data_dir / f"{filename}.{file_format}"
            
            if isinstance(data, pd.DataFrame):
                if file_format == 'parquet':
                    data.to_parquet(filepath, **kwargs)
                elif file_format == 'csv':
                    data.to_csv(filepath, **kwargs)
                else:
                    raise ValueError(f"Unsupported file format for DataFrame: {file_format}")
            elif isinstance(data, (dict, list)):
                if file_format != 'json':
                    raise ValueError(f"Dict/List data can only be saved as JSON, got {file_format}")
                with open(filepath, 'w') as f:
                    json.dump(data, f, **kwargs)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
            logger.info(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
            
    def load_data(
        self,
        filename: str,
        file_format: Optional[str] = None,
        **kwargs
    ) -> Optional[Union[pd.DataFrame, Dict, List]]:
        """
        Load data from disk.
        
        Args:
            filename: Name of the file (with or without extension)
            file_format: Format of the file ('parquet', 'csv', 'json')
            **kwargs: Additional arguments to pass to the load function
            
        Returns:
            Loaded data or None if loading fails
        """
        try:
            # If no format provided, try to detect from filename
            if file_format is None:
                file_format = filename.split('.')[-1] if '.' in filename else 'parquet'
                filename = filename.split('.')[0] if '.' in filename else filename
                
            filepath = self.data_dir / f"{filename}.{file_format}"
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                return None
                
            if file_format == 'parquet':
                return pd.read_parquet(filepath, **kwargs)
            elif file_format == 'csv':
                return pd.read_csv(filepath, **kwargs)
            elif file_format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
            
    def get_latest_file(self, pattern: str = "*") -> Optional[Path]:
        """
        Get the most recent file matching the pattern.
        
        Args:
            pattern: File pattern to match (e.g., "*.parquet")
            
        Returns:
            Path to the latest file or None if no files match
        """
        try:
            files = list(self.data_dir.glob(pattern))
            if not files:
                return None
            return max(files, key=os.path.getmtime)
        except Exception as e:
            logger.error(f"Error finding latest file: {e}")
            return None

    def clean_old_files(self, pattern: str = "*", max_age_days: int = 30) -> int:
        """
        Delete files older than max_age_days.
        
        Args:
            pattern: File pattern to match
            max_age_days: Maximum age of files to keep in days
            
        Returns:
            Number of files deleted
        """
        try:
            cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            deleted = 0
            
            for filepath in self.data_dir.glob(pattern):
                if filepath.is_file() and filepath.stat().st_mtime < cutoff:
                    filepath.unlink()
                    deleted += 1
                    
            logger.info(f"Deleted {deleted} old files")
            return deleted
            
        except Exception as e:
            logger.error(f"Error cleaning old files: {e}")
            return 0