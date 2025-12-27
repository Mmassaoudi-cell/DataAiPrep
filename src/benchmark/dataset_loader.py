"""
Benchmark Dataset Loader for DataAiPrep

Provides utilities to load and work with benchmark datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def list_available_benchmarks(benchmark_path: Optional[Path] = None) -> List[str]:
    """
    List all available benchmark datasets.
    
    Args:
        benchmark_path: Path to benchmark datasets directory
        
    Returns:
        List of available dataset names
    """
    if benchmark_path is None:
        benchmark_path = Path(__file__).parent.parent.parent / 'benchmark_datasets'
    
    benchmark_path = Path(benchmark_path)
    
    if not benchmark_path.exists():
        return []
    
    datasets = []
    for csv_file in benchmark_path.glob('benchmark_*.csv'):
        datasets.append(csv_file.stem)
    
    return sorted(datasets)


def load_benchmark_dataset(
    name: str,
    benchmark_path: Optional[Path] = None,
    include_metadata: bool = True
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Load a benchmark dataset by name.
    
    Args:
        name: Name of the benchmark dataset
        benchmark_path: Path to benchmark datasets directory
        include_metadata: Whether to load and return metadata
        
    Returns:
        Tuple of (DataFrame, metadata_dict) or (DataFrame, None)
    """
    if benchmark_path is None:
        benchmark_path = Path(__file__).parent.parent.parent / 'benchmark_datasets'
    
    benchmark_path = Path(benchmark_path)
    
    # Load data
    data_file = benchmark_path / f"{name}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Benchmark dataset '{name}' not found at {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Load metadata if requested
    metadata = None
    if include_metadata:
        metadata_file = benchmark_path / f"{name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
    
    return df, metadata


def validate_against_benchmark(
    tool_results: Dict,
    dataset_name: str,
    benchmark_path: Optional[Path] = None
) -> Dict:
    """
    Validate tool results against benchmark ground truth.
    
    Args:
        tool_results: Dictionary of detected issues from a data quality tool
        dataset_name: Name of the benchmark dataset
        benchmark_path: Path to benchmark datasets directory
        
    Returns:
        Dictionary with validation metrics
    """
    _, metadata = load_benchmark_dataset(dataset_name, benchmark_path)
    
    if metadata is None:
        raise ValueError(f"No metadata found for benchmark '{dataset_name}'")
    
    ground_truth = metadata.get('ground_truth', {})
    expected_detection = metadata.get('expected_detection', {})
    
    # Calculate detection metrics
    validation_results = {
        'dataset': dataset_name,
        'issue_type': metadata.get('issue_type'),
        'difficulty': metadata.get('difficulty'),
        'expected_detection_rates': expected_detection,
        'actual_detection': {},
        'metrics': {}
    }
    
    # Compare detected issues with ground truth
    # This would be customized based on the specific tool's output format
    
    return validation_results

