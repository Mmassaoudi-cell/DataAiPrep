#!/usr/bin/env python3
"""
Advanced Feature Selection Example

This example demonstrates how to use DataAiPrep's advanced feature selection
capabilities with multiple algorithms and consensus voting.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced import AdvancedFeatureSelector


def create_feature_selection_data():
    """Create a dataset for feature selection demonstration."""
    np.random.seed(42)
    n_samples = 500
    
    # Create informative features
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)
    
    # Create target based on informative features
    y = (2 * X1 + 1.5 * X2 - X3 + np.random.normal(0, 0.5, n_samples)) > 0
    y = y.astype(int)
    
    # Create noise features
    noise_features = {
        f'noise_{i}': np.random.normal(0, 1, n_samples)
        for i in range(1, 8)
    }
    
    # Create redundant features (correlated with informative ones)
    redundant_features = {
        'redundant_1': X1 + np.random.normal(0, 0.1, n_samples),
        'redundant_2': X2 * 1.5 + np.random.normal(0, 0.1, n_samples),
    }
    
    # Combine all features
    data = {
        'informative_1': X1,
        'informative_2': X2,
        'informative_3': X3,
        **noise_features,
        **redundant_features,
        'target': y
    }
    
    return pd.DataFrame(data)


def main():
    """Run advanced feature selection example."""
    print("=" * 60)
    print("DataAiPrep - Advanced Feature Selection Example")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample dataset...")
    df = create_feature_selection_data()
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(X.columns)}")
    print(f"   - Informative: 3 (informative_1, informative_2, informative_3)")
    print(f"   - Noise: 7 (noise_1 to noise_7)")
    print(f"   - Redundant: 2 (redundant_1, redundant_2)")
    
    # Initialize feature selector
    print("\n" + "-" * 60)
    print("🎯 RUNNING FEATURE SELECTION")
    print("-" * 60)
    
    selector = AdvancedFeatureSelector(random_state=42)
    
    # Run multiple selection methods
    print("\n   Running methods: variance, correlation, rfe, lasso")
    
    results = selector.select_features(
        X, y,
        methods=['variance', 'correlation', 'rfe', 'lasso'],
        n_features=5,
        consensus_threshold=0.5
    )
    
    # Display results by method
    print("\n" + "-" * 60)
    print("📋 RESULTS BY METHOD")
    print("-" * 60)
    
    for method, selected in results['method_results'].items():
        print(f"\n   {method.upper()}:")
        print(f"   Selected features: {selected}")
    
    # Display consensus features
    print("\n" + "-" * 60)
    print("🏆 CONSENSUS FEATURES")
    print("-" * 60)
    
    consensus = results['consensus_features']
    print(f"\n   Features selected by multiple methods: {consensus}")
    
    # Display feature rankings
    print("\n" + "-" * 60)
    print("📊 FEATURE RANKINGS")
    print("-" * 60)
    
    print("\n   Rank | Feature | Score")
    print("   " + "-" * 35)
    
    for i, (feat, score) in enumerate(results['feature_rankings'].items(), 1):
        print(f"   {i:4d} | {feat:20s} | {score:.4f}")
    
    # Get optimal features
    optimal = selector.get_optimal_features(min_methods=2)
    print(f"\n   Optimal features (selected by ≥2 methods): {optimal}")
    
    # Generate report
    print("\n" + "-" * 60)
    print("📄 FULL REPORT")
    print("-" * 60)
    print(selector.generate_report())
    
    print("\n" + "=" * 60)
    print("Feature selection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

