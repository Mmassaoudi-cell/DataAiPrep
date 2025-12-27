#!/usr/bin/env python3
"""
Advanced Leakage Detection Example

This example demonstrates how to use DataAiPrep to detect various types of
data leakage between train and test sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced import AdvancedLeakageDetector


def create_leakage_data():
    """Create train/test datasets with intentional leakage for demonstration."""
    np.random.seed(42)
    
    # Create training data
    n_train = 800
    train_data = {
        'customer_id': range(1, n_train + 1),
        'feature_1': np.random.normal(0, 1, n_train),
        'feature_2': np.random.normal(5, 2, n_train),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_train),
        'target': np.random.binomial(1, 0.3, n_train)
    }
    train_df = pd.DataFrame(train_data)
    
    # Create test data
    n_test = 200
    test_data = {
        'customer_id': range(n_train + 1, n_train + n_test + 1),
        'feature_1': np.random.normal(0, 1, n_test),
        'feature_2': np.random.normal(5, 2, n_test),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_test),
        'target': np.random.binomial(1, 0.3, n_test)
    }
    test_df = pd.DataFrame(test_data)
    
    # ===== INTRODUCE LEAKAGE =====
    
    # 1. Train-test contamination: Copy some train rows to test
    contamination_idx = np.random.choice(n_train, size=15, replace=False)
    contaminated_rows = train_df.iloc[contamination_idx].copy()
    contaminated_rows['customer_id'] = range(n_train + n_test + 1, n_train + n_test + 16)
    test_df = pd.concat([test_df, contaminated_rows], ignore_index=True)
    
    # 2. Group leakage: Same customers in train and test
    overlap_idx = np.random.choice(n_test, size=10, replace=False)
    test_df.loc[overlap_idx, 'customer_id'] = np.random.choice(
        range(1, n_train + 1), size=10, replace=False
    )
    
    # 3. Near-duplicates: Slightly modified versions
    near_dup_idx = np.random.choice(n_train, size=5, replace=False)
    near_dups = train_df.iloc[near_dup_idx].copy()
    near_dups['feature_1'] = near_dups['feature_1'] + np.random.normal(0, 0.01, 5)
    near_dups['customer_id'] = range(n_train + n_test + 20, n_train + n_test + 25)
    test_df = pd.concat([test_df, near_dups], ignore_index=True)
    
    return train_df, test_df


def main():
    """Run advanced leakage detection example."""
    print("=" * 60)
    print("DataAiPrep - Advanced Leakage Detection Example")
    print("=" * 60)
    
    # Create sample data with leakage
    print("\n📊 Creating train/test datasets with intentional leakage...")
    train_df, test_df = create_leakage_data()
    
    print(f"   Train set: {len(train_df)} rows")
    print(f"   Test set: {len(test_df)} rows")
    print("\n   Introduced leakage types:")
    print("   - Train-test contamination (exact duplicates)")
    print("   - Group leakage (overlapping customer_ids)")
    print("   - Near-duplicates (slightly modified rows)")
    
    # Initialize detector
    print("\n" + "-" * 60)
    print("🔍 RUNNING LEAKAGE DETECTION")
    print("-" * 60)
    
    detector = AdvancedLeakageDetector(
        similarity_threshold=0.95,
        correlation_threshold=0.95
    )
    
    # Detect all types of leakage
    results = detector.detect_all(
        train_data=train_df,
        test_data=test_df,
        target_column='target',
        entity_column='customer_id'
    )
    
    # Display results
    print("\n" + "-" * 60)
    print("📋 DETECTION RESULTS")
    print("-" * 60)
    
    summary = results['summary']
    print(f"\n   Total issues found: {summary['total_issues']}")
    print(f"   Severity: {summary['severity']}")
    
    # Train-test contamination
    print("\n   🔴 Train-Test Contamination:")
    contamination = results['train_test_contamination']
    print(f"      Detected: {'Yes' if summary['has_contamination'] else 'No'}")
    if contamination.get('exact_duplicates'):
        print(f"      Exact duplicates: {contamination['exact_duplicates']}")
    
    # Near-duplicates
    print("\n   🟠 Near-Duplicates:")
    near_dups = results['near_duplicates']
    print(f"      Detected: {'Yes' if summary['has_near_duplicates'] else 'No'}")
    if near_dups.get('count'):
        print(f"      Count: {near_dups['count']}")
    
    # Group leakage
    print("\n   🟡 Group/Entity Leakage:")
    group_leak = results['group_leakage']
    print(f"      Detected: {'Yes' if summary['has_group_leakage'] else 'No'}")
    if group_leak.get('overlap_count'):
        print(f"      Overlapping entities: {group_leak['overlap_count']}")
        print(f"      Overlap percentage: {group_leak['overlap_percentage']:.2f}%")
    
    # Target leakage
    print("\n   🟢 Target Leakage:")
    target_leak = results['target_leakage']
    print(f"      Detected: {'Yes' if summary['has_target_leakage'] else 'No'}")
    if target_leak.get('high_correlation_features'):
        print(f"      Suspicious features: {target_leak['high_correlation_features']}")
    
    # Generate report
    print("\n" + "-" * 60)
    print("📄 FULL REPORT")
    print("-" * 60)
    print(detector.generate_report())
    
    # Recommendations
    print("\n" + "-" * 60)
    print("💡 RECOMMENDATIONS")
    print("-" * 60)
    
    if summary['has_contamination']:
        print("\n   ⚠️ Remove exact duplicate rows from test set")
    
    if summary['has_near_duplicates']:
        print("\n   ⚠️ Review near-duplicate samples and consider removal")
    
    if summary['has_group_leakage']:
        print("\n   ⚠️ Ensure entity-level split (same customer_id should be in")
        print("      either train OR test, not both)")
    
    if summary['has_target_leakage']:
        print("\n   ⚠️ Remove or investigate features with high target correlation")
    
    print("\n" + "=" * 60)
    print("Leakage detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

