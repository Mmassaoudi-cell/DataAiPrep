#!/usr/bin/env python3
"""
Basic Data Quality Analysis Example

This example demonstrates how to use DataAiPrep for basic data quality assessment.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced import (
    DataQualityPipeline,
    AdvancedMissingAnalyzer,
    EnsembleOutlierDetector
)


def create_sample_data():
    """Create a sample dataset with various quality issues."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_idx = np.random.choice(n_samples, size=50, replace=False)
    df.loc[missing_idx, 'income'] = np.nan
    
    missing_idx2 = np.random.choice(n_samples, size=30, replace=False)
    df.loc[missing_idx2, 'age'] = np.nan
    
    # Introduce outliers
    outlier_idx = np.random.choice(n_samples, size=20, replace=False)
    df.loc[outlier_idx, 'income'] = df['income'].mean() + 5 * df['income'].std()
    
    return df


def main():
    """Run basic data quality analysis."""
    print("=" * 60)
    print("DataAiPrep - Basic Data Quality Analysis Example")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample dataset...")
    df = create_sample_data()
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # 1. Missing Value Analysis
    print("\n" + "-" * 60)
    print("1️⃣ MISSING VALUE ANALYSIS")
    print("-" * 60)
    
    missing_analyzer = AdvancedMissingAnalyzer()
    missing_results = missing_analyzer.analyze(df)
    
    print(f"\n   Total missing: {missing_results['summary']['total_missing']} values "
          f"({missing_results['summary']['total_missing_percentage']:.2f}%)")
    
    if missing_results['summary'].get('missing_by_column'):
        print("\n   Missing by column:")
        for col, info in missing_results['summary']['missing_by_column'].items():
            if info['count'] > 0:
                print(f"   - {col}: {info['count']} ({info['percentage']:.2f}%)")
    
    # 2. Outlier Detection
    print("\n" + "-" * 60)
    print("2️⃣ OUTLIER DETECTION")
    print("-" * 60)
    
    outlier_detector = EnsembleOutlierDetector(
        methods=['iqr', 'zscore', 'isolation_forest']
    )
    outlier_results = outlier_detector.detect(df)
    
    print(f"\n   Consensus outliers: {outlier_results['summary']['consensus_outliers_count']} "
          f"({outlier_results['summary']['consensus_outliers_percentage']:.2f}%)")
    
    print("\n   Outliers by method:")
    for method, count in outlier_results['summary']['outliers_by_method'].items():
        print(f"   - {method}: {count}")
    
    # 3. Full Pipeline
    print("\n" + "-" * 60)
    print("3️⃣ FULL PIPELINE ANALYSIS")
    print("-" * 60)
    
    pipeline = DataQualityPipeline(name="BasicAnalysis")
    pipeline.add_step('completeness', threshold=0.95)
    pipeline.add_step('outlier_detection', methods=['iqr', 'zscore', 'isolation_forest'])
    
    pipeline.run(data=df, target='target', verbose=True)
    
    # Get summary
    summary = pipeline.get_summary()
    print(f"\n   Status: {'✅ SUCCESS' if summary['success'] else '❌ FAILED'}")
    print(f"   Execution time: {summary['execution_time']:.2f}s")
    print(f"   Total recommendations: {summary['total_recommendations']}")
    
    # Get recommendations
    recommendations = pipeline.get_recommendations()
    if recommendations:
        print("\n   Top Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            severity = rec.get('severity', 'medium').upper()
            issue = rec.get('issue', rec.get('message', str(rec)))
            print(f"   {i}. [{severity}] {issue[:60]}...")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

