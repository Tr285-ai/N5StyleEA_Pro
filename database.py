# optimizations/database.py
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
import logging
from contextlib import contextmanager
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """Handles database query optimization and connection management."""
    
    def __init__(self, db_url: str, pool_size: int = 5, max_overflow: int = 10):
        """
        Initialize the database optimizer.
        
        Args:
            db_url: Database connection URL
            pool_size: Number of connections to keep in the pool
            max_overflow: Maximum number of connections to create if pool is full
        """
        self.db_url = db_url
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.Session = scoped_session(
            sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        )
        
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def optimize_queries(self) -> None:
        """Apply database optimizations."""
        with self.engine.connect() as conn:
            # Enable WAL mode for SQLite (if using SQLite)
            if 'sqlite' in self.db_url:
                conn.execute(text('PRAGMA journal_mode=WAL'))
                conn.execute(text('PRAGMA synchronous=NORMAL'))
            
            # Common optimizations for other databases
            conn.execute(text('SET SESSION query_cache_type=1'))
            conn.execute(text('SET SESSION innodb_flush_log_at_trx_commit=2'))  # For MySQL
    
    def batch_insert(self, table_name: str, data: List[Dict[str, Any]], 
                    batch_size: int = 1000) -> None:
        """
        Optimized batch insert operation.
        
        Args:
            table_name: Name of the table to insert into
            data: List of dictionaries containing row data
            batch_size: Number of rows to insert in each batch
        """
        with self.engine.connect() as conn:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                if batch:
                    conn.execute(
                        text(f"INSERT INTO {table_name} VALUES {self._format_batch(batch)}")
                    )
    
    @staticmethod
    def _format_batch(batch: List[Dict[str, Any]]) -> str:
        """Format batch data for SQL insertion."""
        if not batch:
            return ""
        columns = batch[0].keys()
        values = []
        for row in batch:
            values.append(f"({', '.join(repr(row[col]) for col in columns)})")
        return ", ".join(values)
    
    def get_optimized_data(self, query: str, params: Optional[Dict] = None, 
                          chunksize: Optional[int] = None) -> pd.DataFrame:
        """
        Execute a query and return results as a DataFrame with optimizations.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            chunksize: If specified, return an iterator yielding chunks of rows
            
        Returns:
            DataFrame with query results
        """
        with self.engine.connect().execution_options(
            stream_results=True,
            max_row_buffer=1000
        ) as conn:
            return pd.read_sql(
                text(query), 
                conn, 
                params=params,
                chunksize=chunksize
            )