# examples/data_management_example.py
import asyncio
import pandas as pd
from data_management import DataManager

async def main():
    # Initialize data manager
    dm = DataManager(base_path="data")
    
    # Create sample data
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=1000, freq="H"),
        "symbol": ["BTC/USDT"] * 1000,
        "price": [50000 + i * 0.1 for i in range(1000)],
        "volume": [100 + i % 10 for i in range(1000)],
        "exchange": ["binance"] * 1000
    }
    
    # Save dataset with versioning and sharding
    print("Saving dataset with versioning and sharding...")
    result = dm.save_dataset(
        "crypto_ohlcv",
        data,
        versioned=True,
        sharded=True,
        shard_size=200,  # 200 rows per shard
        format_type="parquet"
    )
    print(f"Saved dataset: {result}")
    
    # Create a snapshot
    snapshot_id = dm.create_snapshot("crypto_ohlcv", "initial_load")
    print(f"Created snapshot: {snapshot_id}")
    
    # List snapshots
    snapshots = dm.list_snapshots("crypto_ohlcv")
    print("\nAvailable snapshots:")
    for snap in snapshots:
        print(f"  - {snap['snapshot_name']} ({snap['version']})")
    
    # Load the dataset
    print("\nLoading dataset...")
    loaded_data = dm.load_dataset("crypto_ohlcv")
    print(f"Loaded {len(loaded_data)} rows")
    
    # Get dataset info
    info = dm.get_dataset_info("crypto_ohlcv")
    print("\nDataset info:")
    print(f"  - Versions: {len(info.get('versions', []))}")
    print(f"  - Sharded: {info.get('sharded', False)}")
    if info.get('sharded', False):
        print(f"  - Shard count: {info.get('shard_count', 0)}")
    print(f"  - Rows: {info.get('row_count', 0)}")
    print(f"  - Columns: {', '.join(info.get('columns', []))}")
    
    # Optimize storage
    print("\nOptimizing storage...")
    result = dm.optimize_storage("crypto_ohlcv")
    print(f"Optimization result: {result}")

if __name__ == "__main__":
    asyncio.run(main())