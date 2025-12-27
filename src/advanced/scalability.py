"""
Scalability & Performance Module

Features:
- Dask Integration: Out-of-core processing for large datasets
- Chunked Processing: Process data in memory-efficient batches
- Memory Optimization: Automatic dtype downcasting
- Sampling Strategies: Intelligent sampling for analysis
- Memory Profiling: Track memory usage during processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Generator
from dataclasses import dataclass
import os
import gc
import sys
import warnings
from pathlib import Path
from functools import wraps
import time

warnings.filterwarnings('ignore')


@dataclass
class MemoryStats:
    """Container for memory statistics"""
    initial_mb: float
    final_mb: float
    peak_mb: float
    reduction_percent: float


@dataclass
class ProcessingResult:
    """Container for processing results"""
    success: bool
    chunks_processed: int
    total_rows: int
    processing_time: float
    memory_stats: Optional[MemoryStats]


class LargeDataProcessor:
    """
    Process large datasets that don't fit in memory.
    
    Features:
    - Dask integration for distributed computing
    - Chunked CSV/Parquet reading
    - Memory-efficient processing
    - Progress tracking
    - Automatic dtype optimization
    """
    
    def __init__(
        self,
        chunk_size: int = 100000,
        use_dask: bool = True,
        n_workers: int = None,
        memory_limit: str = "4GB"
    ):
        """
        Initialize the large data processor.
        
        Args:
            chunk_size: Number of rows per chunk
            use_dask: Whether to use Dask if available
            n_workers: Number of Dask workers
            memory_limit: Memory limit per worker
        """
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.n_workers = n_workers or os.cpu_count()
        self.memory_limit = memory_limit
        self._dask_available = self._check_dask()
        self._client = None
        
    def _check_dask(self) -> bool:
        """Check if Dask is available"""
        try:
            import dask
            import dask.dataframe as dd
            self._dask = dask
            self._dd = dd
            return True
        except ImportError:
            return False
    
    def setup_dask_cluster(self) -> bool:
        """
        Set up a local Dask cluster for distributed processing.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._dask_available:
            return False
        
        try:
            from dask.distributed import Client, LocalCluster
            
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit=self.memory_limit
            )
            self._client = Client(cluster)
            return True
        except Exception as e:
            print(f"Failed to set up Dask cluster: {e}")
            return False
    
    def shutdown_dask(self):
        """Shutdown Dask cluster"""
        if self._client is not None:
            self._client.close()
            self._client = None
    
    def read_large_csv(
        self,
        filepath: str,
        columns: List[str] = None,
        dtype: Dict[str, str] = None,
        parse_dates: List[str] = None,
        optimize_memory: bool = True
    ) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Read a large CSV file efficiently.
        
        Args:
            filepath: Path to CSV file
            columns: Columns to read (None for all)
            dtype: Column dtypes
            parse_dates: Columns to parse as dates
            optimize_memory: Whether to optimize memory usage
            
        Returns:
            DataFrame (pandas or Dask)
        """
        if self.use_dask and self._dask_available:
            # Use Dask for lazy loading
            df = self._dd.read_csv(
                filepath,
                usecols=columns,
                dtype=dtype,
                parse_dates=parse_dates,
                blocksize=f"{self.chunk_size * 100}B"
            )
            return df
        else:
            # Use pandas with chunked reading
            chunks = []
            for chunk in pd.read_csv(
                filepath,
                usecols=columns,
                dtype=dtype,
                parse_dates=parse_dates,
                chunksize=self.chunk_size
            ):
                if optimize_memory:
                    chunk = self.optimize_memory(chunk)
                chunks.append(chunk)
            
            return pd.concat(chunks, ignore_index=True)
    
    def read_large_parquet(
        self,
        filepath: str,
        columns: List[str] = None,
        filters: List[Tuple] = None
    ) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Read a large Parquet file efficiently.
        
        Args:
            filepath: Path to Parquet file
            columns: Columns to read
            filters: Row filters (for predicate pushdown)
            
        Returns:
            DataFrame (pandas or Dask)
        """
        if self.use_dask and self._dask_available:
            df = self._dd.read_parquet(
                filepath,
                columns=columns,
                filters=filters
            )
            return df
        else:
            return pd.read_parquet(
                filepath,
                columns=columns,
                filters=filters
            )
    
    def process_in_chunks(
        self,
        filepath: str,
        process_func: Callable[[pd.DataFrame], Any],
        aggregate_func: Callable[[List[Any]], Any] = None,
        progress_callback: Callable[[int, int], None] = None
    ) -> Any:
        """
        Process a large file in chunks.
        
        Args:
            filepath: Path to data file
            process_func: Function to apply to each chunk
            aggregate_func: Function to aggregate results (default: list)
            progress_callback: Callback for progress updates
            
        Returns:
            Aggregated results
        """
        results = []
        total_rows = 0
        chunk_count = 0
        
        # Determine file type
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext == '.csv':
            reader = pd.read_csv(filepath, chunksize=self.chunk_size)
        elif file_ext in ['.parquet', '.pq']:
            # For parquet, load and yield chunks
            full_df = pd.read_parquet(filepath)
            reader = self._chunk_dataframe(full_df)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        for chunk in reader:
            chunk_result = process_func(chunk)
            results.append(chunk_result)
            
            total_rows += len(chunk)
            chunk_count += 1
            
            if progress_callback:
                progress_callback(chunk_count, total_rows)
            
            # Force garbage collection
            gc.collect()
        
        if aggregate_func:
            return aggregate_func(results)
        return results
    
    def _chunk_dataframe(self, df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """Yield chunks from a DataFrame"""
        for start in range(0, len(df), self.chunk_size):
            yield df.iloc[start:start + self.chunk_size]
    
    def parallel_apply(
        self,
        df: pd.DataFrame,
        func: Callable,
        axis: int = 0
    ) -> pd.DataFrame:
        """
        Apply a function in parallel using Dask.
        
        Args:
            df: DataFrame to process
            func: Function to apply
            axis: Axis to apply along (0=rows, 1=columns)
            
        Returns:
            Processed DataFrame
        """
        if self.use_dask and self._dask_available:
            ddf = self._dd.from_pandas(df, npartitions=self.n_workers)
            result = ddf.apply(func, axis=axis, meta=df).compute()
            return result
        else:
            return df.apply(func, axis=axis)
    
    def optimize_memory(
        self,
        df: pd.DataFrame,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting dtypes.
        
        Args:
            df: DataFrame to optimize
            verbose: Whether to print optimization stats
            
        Returns:
            Optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024 ** 2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if cardinality is low
                num_unique = df[col].nunique()
                num_total = len(df[col])
                
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
                    
            elif col_type == 'float64':
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
                
            elif col_type == 'int64':
                # Downcast integers
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 ** 2
        reduction = (1 - final_memory / initial_memory) * 100 if initial_memory > 0 else 0
        
        if verbose:
            print(f"Memory: {initial_memory:.2f} MB → {final_memory:.2f} MB ({reduction:.1f}% reduction)")
        
        return df
    
    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed memory usage statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with memory statistics
        """
        memory_by_col = df.memory_usage(deep=True)
        total_memory = memory_by_col.sum()
        
        return {
            'total_mb': total_memory / 1024 ** 2,
            'by_column': {
                col: mem / 1024 ** 2
                for col, mem in memory_by_col.items()
            },
            'by_dtype': {
                str(dtype): df.select_dtypes(include=[dtype]).memory_usage(deep=True).sum() / 1024 ** 2
                for dtype in df.dtypes.unique()
            },
            'shape': df.shape,
            'bytes_per_row': total_memory / len(df) if len(df) > 0 else 0
        }


class IntelligentSampler:
    """
    Intelligent sampling strategies for large dataset analysis.
    
    Strategies:
    - Random sampling
    - Stratified sampling
    - Systematic sampling
    - Reservoir sampling (for streaming)
    - Cluster-based sampling
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def random_sample(
        self,
        df: pd.DataFrame,
        n: int = None,
        frac: float = None
    ) -> pd.DataFrame:
        """
        Simple random sampling.
        
        Args:
            df: DataFrame to sample from
            n: Number of samples
            frac: Fraction of samples
            
        Returns:
            Sampled DataFrame
        """
        return df.sample(n=n, frac=frac, random_state=self.random_state)
    
    def stratified_sample(
        self,
        df: pd.DataFrame,
        stratify_col: str,
        n: int = None,
        frac: float = None
    ) -> pd.DataFrame:
        """
        Stratified sampling maintaining class proportions.
        
        Args:
            df: DataFrame to sample from
            stratify_col: Column to stratify by
            n: Total number of samples
            frac: Fraction of samples
            
        Returns:
            Sampled DataFrame
        """
        if stratify_col not in df.columns:
            return self.random_sample(df, n=n, frac=frac)
        
        if n is not None:
            frac = n / len(df)
        elif frac is None:
            frac = 0.1
        
        # Sample from each stratum
        samples = []
        for value in df[stratify_col].unique():
            stratum = df[df[stratify_col] == value]
            stratum_n = max(1, int(len(stratum) * frac))
            
            if len(stratum) > 0:
                sample = stratum.sample(
                    n=min(stratum_n, len(stratum)),
                    random_state=self.random_state
                )
                samples.append(sample)
        
        return pd.concat(samples, ignore_index=True)
    
    def systematic_sample(
        self,
        df: pd.DataFrame,
        step: int = None,
        n: int = None
    ) -> pd.DataFrame:
        """
        Systematic sampling (every kth element).
        
        Args:
            df: DataFrame to sample from
            step: Sampling interval
            n: Approximate number of samples
            
        Returns:
            Sampled DataFrame
        """
        if step is None and n is not None:
            step = max(1, len(df) // n)
        elif step is None:
            step = 10
        
        start = np.random.randint(0, step)
        indices = range(start, len(df), step)
        
        return df.iloc[list(indices)]
    
    def reservoir_sample(
        self,
        iterator: Generator,
        k: int
    ) -> List[Any]:
        """
        Reservoir sampling for streaming data.
        
        Maintains a random sample of k items from a stream
        without knowing the total size in advance.
        
        Args:
            iterator: Iterator over items
            k: Sample size to maintain
            
        Returns:
            List of k sampled items
        """
        reservoir = []
        
        for i, item in enumerate(iterator):
            if i < k:
                reservoir.append(item)
            else:
                # Replace with decreasing probability
                j = np.random.randint(0, i + 1)
                if j < k:
                    reservoir[j] = item
        
        return reservoir
    
    def cluster_sample(
        self,
        df: pd.DataFrame,
        cluster_col: str,
        n_clusters: int = None,
        frac_clusters: float = None
    ) -> pd.DataFrame:
        """
        Cluster-based sampling (select entire clusters).
        
        Args:
            df: DataFrame to sample from
            cluster_col: Column defining clusters
            n_clusters: Number of clusters to sample
            frac_clusters: Fraction of clusters to sample
            
        Returns:
            Sampled DataFrame
        """
        clusters = df[cluster_col].unique()
        
        if n_clusters is None and frac_clusters is not None:
            n_clusters = int(len(clusters) * frac_clusters)
        elif n_clusters is None:
            n_clusters = min(10, len(clusters))
        
        selected_clusters = np.random.choice(
            clusters,
            size=min(n_clusters, len(clusters)),
            replace=False
        )
        
        return df[df[cluster_col].isin(selected_clusters)]
    
    def importance_sample(
        self,
        df: pd.DataFrame,
        weight_col: str,
        n: int
    ) -> pd.DataFrame:
        """
        Importance sampling based on a weight column.
        
        Args:
            df: DataFrame to sample from
            weight_col: Column with sampling weights
            n: Number of samples
            
        Returns:
            Sampled DataFrame
        """
        weights = df[weight_col].values
        weights = weights / weights.sum()  # Normalize
        
        indices = np.random.choice(
            len(df),
            size=min(n, len(df)),
            replace=False,
            p=weights
        )
        
        return df.iloc[indices]


class MemoryProfiler:
    """
    Profile memory usage during data processing.
    
    Features:
    - Track memory usage over time
    - Identify memory-intensive operations
    - Generate memory reports
    """
    
    def __init__(self):
        """Initialize the memory profiler"""
        self._snapshots = []
        self._start_time = None
    
    def start(self):
        """Start memory profiling"""
        self._start_time = time.time()
        self._take_snapshot("start")
    
    def checkpoint(self, label: str):
        """Take a memory snapshot with a label"""
        self._take_snapshot(label)
    
    def _take_snapshot(self, label: str):
        """Record current memory state"""
        import tracemalloc
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        current, peak = tracemalloc.get_traced_memory()
        
        self._snapshots.append({
            'label': label,
            'time': time.time() - (self._start_time or time.time()),
            'current_mb': current / 1024 ** 2,
            'peak_mb': peak / 1024 ** 2
        })
    
    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return report"""
        import tracemalloc
        
        self._take_snapshot("end")
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        if not self._snapshots:
            return {'error': 'No snapshots recorded'}
        
        return {
            'snapshots': self._snapshots,
            'total_time': self._snapshots[-1]['time'],
            'peak_memory_mb': max(s['peak_mb'] for s in self._snapshots),
            'final_memory_mb': self._snapshots[-1]['current_mb']
        }
    
    @staticmethod
    def profile_function(func: Callable) -> Callable:
        """Decorator to profile a function's memory usage"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import tracemalloc
            
            tracemalloc.start()
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"[{func.__name__}] Time: {time.time() - start_time:.2f}s, "
                  f"Peak Memory: {peak / 1024 ** 2:.2f} MB")
            
            return result
        
        return wrapper


def get_system_memory_info() -> Dict[str, float]:
    """Get current system memory information"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / 1024 ** 3,
            'available_gb': mem.available / 1024 ** 3,
            'used_gb': mem.used / 1024 ** 3,
            'percent_used': mem.percent
        }
    except ImportError:
        return {
            'note': 'psutil not installed, cannot get system memory info'
        }

