# data_management/storage.py
import os
from typing import Any, Dict, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ColumnarStorage:
    """Handles efficient columnar storage using Parquet and Arrow formats."""
    
    def __init__(self, base_path: str = "data/storage"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def _get_file_path(self, dataset: str, format_type: str = "parquet") -> str:
        """Get the full path for a dataset file."""
        return os.path.join(self.base_path, f"{dataset}.{format_type}")
    
    def save(self, dataset: str, data: Any, format_type: str = "parquet", **kwargs) -> str:
        """
        Save data in columnar format.
        
        Args:
            dataset: Name of the dataset
            data: Data to save (can be dict, list, pandas DataFrame, etc.)
            format_type: 'parquet' or 'feather'
            **kwargs: Additional arguments for the writer
            
        Returns:
            Path to the saved file
        """
        import pandas as pd
        
        # Convert to pandas DataFrame if needed
        if not isinstance(data, (pd.DataFrame, pa.Table)):
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported data type. Must be dict, list, pandas DataFrame, or pyarrow Table.")
        
        # Convert to Arrow Table if not already
        if isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data)
        else:
            table = data
            
        # Save in requested format
        file_path = self._get_file_path(dataset, format_type)
        
        if format_type == "parquet":
            pq.write_table(table, file_path, **kwargs)
        elif format_type == "feather":
            feather.write_feather(table, file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        return file_path
    
    def load(self, dataset: str, format_type: str = "parquet", **kwargs) -> pa.Table:
        """
        Load data from columnar storage.
        
        Args:
            dataset: Name of the dataset
            format_type: 'parquet' or 'feather'
            **kwargs: Additional arguments for the reader
            
        Returns:
            PyArrow Table containing the data
        """
        file_path = self._get_file_path(dataset, format_type)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset {dataset} not found at {file_path}")
            
        if format_type == "parquet":
            return pq.read_table(file_path, **kwargs)
        elif format_type == "feather":
            return feather.read_table(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def to_pandas(self, dataset: str, **kwargs) -> 'pd.DataFrame':
        """Load data as a pandas DataFrame."""
        table = self.load(dataset, **kwargs)
        return table.to_pandas()