"""
DataAiPrep Benchmark Dataset Package

Provides standardized benchmark datasets with known data quality issues
for validating data quality assessment tools.

Usage:
    from src.benchmark import BenchmarkDatasetGenerator
    
    generator = BenchmarkDatasetGenerator()
    datasets = generator.generate_all_benchmarks()
"""

from .dataset_generator import BenchmarkDatasetGenerator
from .dataset_loader import load_benchmark_dataset, list_available_benchmarks

__all__ = [
    'BenchmarkDatasetGenerator',
    'load_benchmark_dataset',
    'list_available_benchmarks'
]

