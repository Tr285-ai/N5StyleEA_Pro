# data_management/sharding.py
import os
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Dict, List, Optional, Union, Callable
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataSharding:
    """Handles horizontal and vertical data sharding for large datasets."""
    
    def __init__(self, base_path: str = "data/sharded"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def _get_shard_path(self, dataset: str, shard_id: Union[int, str]) -> str:
        """Get the path for a specific shard."""
        shard_dir = os.path.join(self.base_path, dataset)
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"shard_{shard_id}.parquet")
    
    def shard_by_rows(
        self,
        dataset: str,
        data: Any,
        shard_size: int = 10000,
        shard_key: Optional[Callable] = None
    ) -> List[str]:
        """
        Shard a dataset by rows.
        
        Args:
            dataset: Name of the dataset
            data: Input data (pandas DataFrame or Arrow Table)
            shard_size: Maximum number of rows per shard
            shard_key: Optional function to determine shard for each row
            
        Returns:
            List of paths to created shards
        """
        import pandas as pd
        
        # Convert to pandas if needed
        if isinstance(data, pa.Table):
            df = data.to_pandas()
        else:
            df = data.copy()
            
        shard_paths = []
        total_rows = len(df)
        
        if shard_key is not None:
            # Shard using custom sharding function
            df['_shard'] = df.apply(shard_key, axis=1)
            for shard_id, group in df.groupby('_shard'):
                shard_path = self._save_shard(dataset, shard_id, group.drop('_shard', axis=1))
                shard_paths.append(shard_path)
        else:
            # Simple row-based sharding
            num_shards = (total_rows + shard_size - 1) // shard_size
            for i in range(num_shards):
                start_idx = i * shard_size
                end_idx = min((i + 1) * shard_size, total_rows)
                shard_data = df.iloc[start_idx:end_idx]
                shard_path = self._save_shard(dataset, i, shard_data)
                shard_paths.append(shard_path)
                
        return shard_paths
    
    def _save_shard(self, dataset: str, shard_id: Union[int, str], data: Any) -> str:
        """Save a single shard to disk."""
        shard_path = self._get_shard_path(dataset, shard_id)
        
        if isinstance(data, pa.Table):
            table = data
        else:
            import pandas as pd
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            table = pa.Table.from_pandas(data)
            
        pq.write_table(table, shard_path)
        return shard_path
    
    def load_shard(self, dataset: str, shard_id: Union[int, str]) -> pa.Table:
        """Load a specific shard."""
        shard_path = self._get_shard_path(dataset, shard_id)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard {shard_id} not found for dataset {dataset}")
        return pq.read_table(shard_path)
    
    def load_all_shards(self, dataset: str) -> pa.Table:
        """Load and combine all shards for a dataset."""
        import glob
        shard_dir = os.path.join(self.base_path, dataset)
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))
        
        if not shard_files:
            raise FileNotFoundError(f"No shards found for dataset {dataset}")
            
        tables = [pq.read_table(f) for f in shard_files]
        return pa.concat_tables(tables)
    
    def shard_by_columns(
        self,
        dataset: str,
        data: Any,
        column_groups: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """
        Shard a dataset by column groups.
        
        Args:
            dataset: Name of the dataset
            data: Input data (pandas DataFrame or Arrow Table)
            column_groups: Dictionary mapping shard names to column lists
            
        Returns:
            Dictionary mapping shard names to file paths
        """
        import pandas as pd
        
        if isinstance(data, pa.Table):
            df = data.to_pandas()
        else:
            df = data.copy()
            
        shard_paths = {}
        
        for shard_name, columns in column_groups.items():
            # Ensure all requested columns exist
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns not found in data: {missing}")
                
            shard_data = df[columns]
            shard_path = self._save_shard(f"{dataset}_{shard_name}", 0, shard_data)
            shard_paths[shard_name] = shard_path
            
        return shard_paths