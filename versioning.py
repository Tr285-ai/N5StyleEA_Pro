# data_management/versioning.py
import os
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataVersioning:
    """Handles data versioning with support for snapshots and diffs."""
    
    def __init__(self, storage_path: str = "data/versions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def _get_version_path(self, dataset: str, version: str) -> str:
        return os.path.join(self.storage_path, dataset, f"{version}.parquet")
    
    def create_version(self, dataset: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """Create a new version of the dataset."""
        from pyarrow import Table
        import pyarrow.parquet as pq
        
        if metadata is None:
            metadata = {}
            
        # Generate version ID (timestamp + hash of data)
        timestamp = datetime.utcnow().isoformat()
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()[:12]
        version_id = f"{timestamp}_{data_hash}"
        
        # Create dataset directory if it doesn't exist
        dataset_dir = os.path.join(self.storage_path, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Convert data to Arrow table if it isn't already
        if not isinstance(data, Table):
            if isinstance(data, dict):
                import pandas as pd
                data = pd.DataFrame([data])
            table = Table.from_pandas(data)
        else:
            table = data
            
        # Add metadata
        metadata.update({
            "version_id": version_id,
            "created_at": timestamp,
            "parent_version": self.get_latest_version(dataset)
        })
        
        # Write version
        version_path = self._get_version_path(dataset, version_id)
        pq.write_table(table, version_path, metadata_collector=metadata)
        
        # Update latest version pointer
        with open(os.path.join(dataset_dir, "latest"), "w") as f:
            f.write(version_id)
            
        return version_id
    
    def get_version(self, dataset: str, version: str = "latest") -> Any:
        """Retrieve a specific version of a dataset."""
        if version == "latest":
            version = self.get_latest_version(dataset)
            
        version_path = self._get_version_path(dataset, version)
        if not os.path.exists(version_path):
            raise ValueError(f"Version {version} not found for dataset {dataset}")
            
        import pyarrow.parquet as pq
        return pq.read_table(version_path)
    
    def get_latest_version(self, dataset: str) -> Optional[str]:
        """Get the latest version ID for a dataset."""
        latest_file = os.path.join(self.storage_path, dataset, "latest")
        if os.path.exists(latest_file):
            with open(latest_file, "r") as f:
                return f.read().strip()
        return None
    
    def list_versions(self, dataset: str) -> list:
        """List all available versions for a dataset."""
        dataset_dir = os.path.join(self.storage_path, dataset)
        if not os.path.exists(dataset_dir):
            return []
            
        versions = []
        for f in os.listdir(dataset_dir):
            if f.endswith('.parquet'):
                versions.append(f.replace('.parquet', ''))
        return sorted(versions)