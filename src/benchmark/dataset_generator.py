"""
Benchmark Dataset Generator for DataAiPrep

Creates 10 standardized synthetic datasets with known data quality issues
for validating and benchmarking data quality assessment tools.

Based on established benchmarks:
- CleanML Benchmark (Chu-Data-Lab)
- UCI ML Repository documented issues
- Scikit-learn synthetic data generators
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib


@dataclass
class BenchmarkDataset:
    """Container for a benchmark dataset with metadata"""
    name: str
    data: pd.DataFrame
    target_column: str
    known_issues: Dict[str, Any]
    ground_truth: Dict[str, Any]
    description: str
    issue_type: str
    difficulty: str  # 'easy', 'medium', 'hard'
    expected_detection: Dict[str, float]  # Expected detection rates


class BenchmarkDatasetGenerator:
    """
    Generates 10 standardized benchmark datasets with known data quality issues.
    
    Dataset Suite:
    1. MCAR Missing Values - Missing Completely At Random
    2. MAR Missing Values - Missing At Random  
    3. MNAR Missing Values - Missing Not At Random
    4. Perfect Correlations - Data leakage via redundant features
    5. Temporal Leakage - Future information leak in time-series
    6. High Outlier Rates - Extreme outliers affecting distributions
    7. Class Imbalance - Severely imbalanced classification
    8. Duplicate Features - Exact duplicate columns
    9. Target Contamination - Target variable leakage
    10. Mixed Quality Issues - Multiple combined issues
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_all_benchmarks(self, save_path: Optional[Path] = None) -> Dict[str, BenchmarkDataset]:
        """
        Generate all 10 benchmark datasets.
        
        Args:
            save_path: Optional path to save datasets as CSV files
            
        Returns:
            Dictionary mapping dataset names to BenchmarkDataset objects
        """
        datasets = {}
        
        generators = [
            self._generate_mcar_missing,
            self._generate_mar_missing,
            self._generate_mnar_missing,
            self._generate_perfect_correlations,
            self._generate_temporal_leakage,
            self._generate_high_outliers,
            self._generate_class_imbalance,
            self._generate_duplicate_features,
            self._generate_target_contamination,
            self._generate_mixed_issues,
        ]
        
        for generator in generators:
            dataset = generator()
            datasets[dataset.name] = dataset
            
            if save_path:
                self._save_dataset(dataset, save_path)
                
        return datasets
    
    def _generate_mcar_missing(self) -> BenchmarkDataset:
        """Dataset 1: Missing Completely At Random (MCAR)"""
        np.random.seed(self.random_state)
        
        # Generate base data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=0, n_repeated=0, random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        
        # Introduce MCAR missing values (random 10% missing)
        missing_mask = np.random.random(df.shape) < 0.10
        missing_mask[:, -1] = False  # Don't add missing to target
        
        missing_columns = []
        for i, col in enumerate(df.columns[:-1]):
            if missing_mask[:, i].any():
                df.loc[missing_mask[:, i], col] = np.nan
                missing_columns.append(col)
        
        return BenchmarkDataset(
            name="benchmark_01_mcar_missing",
            data=df,
            target_column='target',
            known_issues={
                'missing_type': 'MCAR',
                'missing_rate': 0.10,
                'affected_columns': missing_columns[:10],  # Top 10
                'total_missing_cells': int(missing_mask.sum())
            },
            ground_truth={
                'expected_missing_percentage': 10.0,
                'missingness_pattern': 'random',
                'little_mcar_test_should_pass': True
            },
            description="Dataset with 10% Missing Completely At Random (MCAR) values. "
                       "Missing values are randomly distributed with no pattern.",
            issue_type='missing_data',
            difficulty='easy',
            expected_detection={
                'missing_detection': 0.99,
                'mcar_classification': 0.90
            }
        )
    
    def _generate_mar_missing(self) -> BenchmarkDataset:
        """Dataset 2: Missing At Random (MAR)"""
        np.random.seed(self.random_state + 1)
        
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=0, n_repeated=0, random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        
        # MAR: Missing in feature_5 depends on feature_0 values
        # If feature_0 > median, feature_5 has 30% chance of being missing
        median_f0 = df['feature_0'].median()
        mar_mask = (df['feature_0'] > median_f0) & (np.random.random(len(df)) < 0.30)
        df.loc[mar_mask, 'feature_5'] = np.nan
        
        # Also add MAR pattern to feature_10 based on feature_1
        median_f1 = df['feature_1'].median()
        mar_mask2 = (df['feature_1'] < median_f1) & (np.random.random(len(df)) < 0.25)
        df.loc[mar_mask2, 'feature_10'] = np.nan
        
        return BenchmarkDataset(
            name="benchmark_02_mar_missing",
            data=df,
            target_column='target',
            known_issues={
                'missing_type': 'MAR',
                'dependency_pairs': [
                    ('feature_5', 'feature_0', 'positive'),
                    ('feature_10', 'feature_1', 'negative')
                ],
                'affected_columns': ['feature_5', 'feature_10']
            },
            ground_truth={
                'feature_5_missing_when_f0_high': True,
                'feature_10_missing_when_f1_low': True,
                'correlation_with_observed': True
            },
            description="Dataset with Missing At Random (MAR) values. "
                       "Missingness in feature_5 depends on feature_0 values; "
                       "missingness in feature_10 depends on feature_1 values.",
            issue_type='missing_data',
            difficulty='medium',
            expected_detection={
                'missing_detection': 0.99,
                'mar_classification': 0.85
            }
        )
    
    def _generate_mnar_missing(self) -> BenchmarkDataset:
        """Dataset 3: Missing Not At Random (MNAR)"""
        np.random.seed(self.random_state + 2)
        
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=0, n_repeated=0, random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        
        # MNAR: High values of feature_3 are more likely to be missing
        # (e.g., people with high income less likely to report it)
        p90_f3 = df['feature_3'].quantile(0.90)
        mnar_mask = (df['feature_3'] > p90_f3) & (np.random.random(len(df)) < 0.70)
        original_f3 = df['feature_3'].copy()
        df.loc[mnar_mask, 'feature_3'] = np.nan
        
        # Also: extreme values of feature_8 are missing
        p10_f8 = df['feature_8'].quantile(0.10)
        p90_f8 = df['feature_8'].quantile(0.90)
        mnar_mask2 = ((df['feature_8'] < p10_f8) | (df['feature_8'] > p90_f8)) & \
                     (np.random.random(len(df)) < 0.50)
        df.loc[mnar_mask2, 'feature_8'] = np.nan
        
        return BenchmarkDataset(
            name="benchmark_03_mnar_missing",
            data=df,
            target_column='target',
            known_issues={
                'missing_type': 'MNAR',
                'self_dependency': [
                    ('feature_3', 'high_values_missing'),
                    ('feature_8', 'extreme_values_missing')
                ],
                'affected_columns': ['feature_3', 'feature_8']
            },
            ground_truth={
                'feature_3_high_values_missing': True,
                'feature_8_extremes_missing': True,
                'missingness_depends_on_unobserved': True
            },
            description="Dataset with Missing Not At Random (MNAR) values. "
                       "High values of feature_3 are likely to be missing; "
                       "extreme values of feature_8 are likely to be missing.",
            issue_type='missing_data',
            difficulty='hard',
            expected_detection={
                'missing_detection': 0.99,
                'mnar_classification': 0.75
            }
        )
    
    def _generate_perfect_correlations(self) -> BenchmarkDataset:
        """Dataset 4: Perfect Correlations (Data Leakage)"""
        np.random.seed(self.random_state + 3)
        
        X, y = make_classification(
            n_samples=1000, n_features=15, n_informative=10,
            n_redundant=5, n_repeated=0, random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        df['target'] = y
        
        # Add perfectly correlated features (explicit leakage)
        df['leak_1'] = df['feature_0'] * 1.0  # Exact copy
        df['leak_2'] = df['feature_1'] + 0.001 * np.random.randn(len(df))  # Near-perfect
        df['leak_3'] = df['feature_0'] + df['feature_1']  # Linear combination
        df['leak_4'] = -df['feature_2']  # Negatively correlated
        df['leak_5'] = df['feature_0'] * 2 + 5  # Scaled and shifted
        
        return BenchmarkDataset(
            name="benchmark_04_perfect_correlations",
            data=df,
            target_column='target',
            known_issues={
                'leakage_type': 'perfect_correlation',
                'leaky_pairs': [
                    ('leak_1', 'feature_0', 1.0),
                    ('leak_2', 'feature_1', 0.999),
                    ('leak_4', 'feature_2', -1.0),
                    ('leak_5', 'feature_0', 1.0)
                ],
                'linear_combinations': [('leak_3', ['feature_0', 'feature_1'])]
            },
            ground_truth={
                'num_leaky_features': 5,
                'should_remove': ['leak_1', 'leak_2', 'leak_3', 'leak_4', 'leak_5']
            },
            description="Dataset with perfectly and near-perfectly correlated features "
                       "that represent data leakage scenarios.",
            issue_type='data_leakage',
            difficulty='easy',
            expected_detection={
                'perfect_correlation_detection': 0.98,
                'leakage_identification': 0.95
            }
        )
    
    def _generate_temporal_leakage(self) -> BenchmarkDataset:
        """Dataset 5: Temporal Leakage (Future Information)"""
        np.random.seed(self.random_state + 4)
        
        n_samples = 1000
        
        # Create time-ordered data
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        
        # Base features (available at prediction time)
        df = pd.DataFrame({
            'date': dates,
            'feature_1': np.random.randn(n_samples).cumsum(),  # Random walk
            'feature_2': np.sin(np.arange(n_samples) * 2 * np.pi / 365),  # Seasonal
            'feature_3': np.random.randn(n_samples),
        })
        
        # Target: depends on future values (realistic scenario)
        df['target'] = (df['feature_1'].shift(-7) > df['feature_1']).astype(int)
        
        # LEAKY FEATURES: Use future information
        df['leak_future_avg'] = df['feature_1'].shift(-7).rolling(7, min_periods=1).mean()
        df['leak_future_max'] = df['feature_1'].shift(-3).rolling(3, min_periods=1).max()
        df['leak_next_day'] = df['feature_1'].shift(-1)
        
        # Also add a "same-time" leak (target encoded in feature)
        df['leak_target_encoded'] = df['target'] * 0.9 + np.random.randn(n_samples) * 0.1
        
        # Remove rows with NaN from shift operations
        df = df.dropna().reset_index(drop=True)
        
        return BenchmarkDataset(
            name="benchmark_05_temporal_leakage",
            data=df,
            target_column='target',
            known_issues={
                'leakage_type': 'temporal',
                'future_information_columns': ['leak_future_avg', 'leak_future_max', 'leak_next_day'],
                'target_leakage_columns': ['leak_target_encoded'],
                'temporal_column': 'date'
            },
            ground_truth={
                'future_leak_features': 3,
                'target_leak_features': 1,
                'should_remove': ['leak_future_avg', 'leak_future_max', 'leak_next_day', 'leak_target_encoded']
            },
            description="Time-series dataset with temporal leakage. Features contain "
                       "future information that would not be available at prediction time.",
            issue_type='data_leakage',
            difficulty='medium',
            expected_detection={
                'temporal_leakage_detection': 0.90,
                'target_leakage_detection': 0.95
            }
        )
    
    def _generate_high_outliers(self) -> BenchmarkDataset:
        """Dataset 6: High Outlier Rates"""
        np.random.seed(self.random_state + 5)
        
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=0, n_repeated=0, random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        
        outlier_info = {}
        
        # Add different types of outliers
        # Type 1: Point outliers (extreme values)
        outlier_idx = np.random.choice(len(df), size=50, replace=False)
        df.loc[outlier_idx, 'feature_0'] = df['feature_0'].mean() + 10 * df['feature_0'].std()
        outlier_info['feature_0'] = {'type': 'point', 'count': 50}
        
        # Type 2: Contextual outliers (values that are outliers in context)
        outlier_idx2 = np.random.choice(len(df), size=30, replace=False)
        df.loc[outlier_idx2, 'feature_5'] = -df.loc[outlier_idx2, 'feature_5'] * 5
        outlier_info['feature_5'] = {'type': 'contextual', 'count': 30}
        
        # Type 3: Collective outliers (groups of unusual values)
        df.loc[100:120, 'feature_10'] = 100  # Constant unusual value for a range
        outlier_info['feature_10'] = {'type': 'collective', 'count': 20}
        
        # Type 4: Heavy-tailed distribution
        df['feature_heavy_tail'] = np.random.standard_t(df=3, size=len(df))  # t-distribution
        outlier_info['feature_heavy_tail'] = {'type': 'heavy_tail', 'expected_outliers': 'high'}
        
        return BenchmarkDataset(
            name="benchmark_06_high_outliers",
            data=df,
            target_column='target',
            known_issues={
                'outlier_types': ['point', 'contextual', 'collective', 'heavy_tail'],
                'affected_columns': list(outlier_info.keys()),
                'outlier_details': outlier_info,
                'total_outlier_rate': 0.10
            },
            ground_truth={
                'feature_0_outliers': 50,
                'feature_5_outliers': 30,
                'feature_10_outliers': 20,
                'expected_iqr_outliers': 'high'
            },
            description="Dataset with multiple types of outliers: point outliers, "
                       "contextual outliers, collective outliers, and heavy-tailed distribution.",
            issue_type='outliers',
            difficulty='medium',
            expected_detection={
                'outlier_detection': 0.90,
                'outlier_type_classification': 0.80
            }
        )
    
    def _generate_class_imbalance(self) -> BenchmarkDataset:
        """Dataset 7: Severe Class Imbalance"""
        np.random.seed(self.random_state + 6)
        
        # Create severely imbalanced dataset (95:5 ratio)
        X, y = make_classification(
            n_samples=2000, n_features=20, n_informative=15,
            n_redundant=3, n_repeated=0, n_classes=2,
            weights=[0.95, 0.05],  # 95% class 0, 5% class 1
            flip_y=0.01,
            random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        
        class_counts = df['target'].value_counts().to_dict()
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        
        return BenchmarkDataset(
            name="benchmark_07_class_imbalance",
            data=df,
            target_column='target',
            known_issues={
                'imbalance_type': 'binary',
                'class_distribution': class_counts,
                'imbalance_ratio': imbalance_ratio,
                'minority_class': 1,
                'minority_percentage': class_counts.get(1, 0) / len(df) * 100
            },
            ground_truth={
                'expected_imbalance_ratio': 19.0,  # ~95:5
                'minority_class_percentage': 5.0,
                'recommended_strategies': ['SMOTE', 'class_weight', 'undersampling']
            },
            description="Severely imbalanced binary classification dataset with "
                       "approximately 95:5 class ratio.",
            issue_type='class_imbalance',
            difficulty='easy',
            expected_detection={
                'imbalance_detection': 0.99,
                'ratio_estimation': 0.95
            }
        )
    
    def _generate_duplicate_features(self) -> BenchmarkDataset:
        """Dataset 8: Duplicate and Near-Duplicate Features"""
        np.random.seed(self.random_state + 7)
        
        X, y = make_classification(
            n_samples=1000, n_features=15, n_informative=10,
            n_redundant=0, n_repeated=0, random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        df['target'] = y
        
        # Exact duplicates
        df['dup_exact_1'] = df['feature_0'].copy()
        df['dup_exact_2'] = df['feature_1'].copy()
        
        # Near-duplicates (with tiny noise)
        df['dup_near_1'] = df['feature_2'] + np.random.randn(len(df)) * 1e-10
        df['dup_near_2'] = df['feature_3'] * 1.0000001
        
        # Renamed duplicates (different names, same data)
        df['customer_id_v1'] = df['feature_4'].copy()
        df['customer_id_v2'] = df['feature_4'].copy()
        
        return BenchmarkDataset(
            name="benchmark_08_duplicate_features",
            data=df,
            target_column='target',
            known_issues={
                'duplicate_type': ['exact', 'near_duplicate', 'renamed'],
                'exact_duplicates': [
                    ('dup_exact_1', 'feature_0'),
                    ('dup_exact_2', 'feature_1'),
                    ('customer_id_v1', 'feature_4'),
                    ('customer_id_v2', 'feature_4')
                ],
                'near_duplicates': [
                    ('dup_near_1', 'feature_2'),
                    ('dup_near_2', 'feature_3')
                ]
            },
            ground_truth={
                'num_exact_duplicates': 4,
                'num_near_duplicates': 2,
                'columns_to_remove': ['dup_exact_1', 'dup_exact_2', 'dup_near_1', 
                                     'dup_near_2', 'customer_id_v2']
            },
            description="Dataset with exact duplicate features and near-duplicate features "
                       "that represent redundant information.",
            issue_type='duplicate_features',
            difficulty='easy',
            expected_detection={
                'exact_duplicate_detection': 0.99,
                'near_duplicate_detection': 0.95
            }
        )
    
    def _generate_target_contamination(self) -> BenchmarkDataset:
        """Dataset 9: Target Variable Contamination"""
        np.random.seed(self.random_state + 8)
        
        X, y = make_classification(
            n_samples=1000, n_features=15, n_informative=10,
            n_redundant=0, n_repeated=0, random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        df['target'] = y
        
        # Target contamination patterns
        # 1. Direct leak: feature derived from target
        df['leak_direct'] = df['target'] + np.random.randn(len(df)) * 0.1
        
        # 2. Encoded leak: target one-hot encoded and renamed
        df['status_approved'] = (df['target'] == 1).astype(int)
        df['status_rejected'] = (df['target'] == 0).astype(int)
        
        # 3. Calculated from target: aggregated stats
        df['group_target_mean'] = df.groupby(df.index // 100)['target'].transform('mean')
        
        # 4. Subtle leak: highly correlated with target
        df['leak_subtle'] = df['target'] * 0.8 + df['feature_0'] * 0.2
        
        return BenchmarkDataset(
            name="benchmark_09_target_contamination",
            data=df,
            target_column='target',
            known_issues={
                'contamination_type': 'target_leakage',
                'direct_leaks': ['leak_direct', 'status_approved', 'status_rejected'],
                'calculated_leaks': ['group_target_mean'],
                'subtle_leaks': ['leak_subtle'],
                'all_contaminated': ['leak_direct', 'status_approved', 'status_rejected', 
                                    'group_target_mean', 'leak_subtle']
            },
            ground_truth={
                'num_contaminated_features': 5,
                'correlation_with_target': {
                    'leak_direct': 0.99,
                    'status_approved': 1.0,
                    'status_rejected': -1.0,
                    'leak_subtle': 0.90
                }
            },
            description="Dataset with target variable contamination where features "
                       "are derived from or highly correlated with the target.",
            issue_type='data_leakage',
            difficulty='medium',
            expected_detection={
                'direct_leak_detection': 0.99,
                'subtle_leak_detection': 0.85
            }
        )
    
    def _generate_mixed_issues(self) -> BenchmarkDataset:
        """Dataset 10: Multiple Combined Quality Issues"""
        np.random.seed(self.random_state + 9)
        
        X, y = make_classification(
            n_samples=1500, n_features=25, n_informative=15,
            n_redundant=5, n_repeated=0, n_classes=2,
            weights=[0.80, 0.20],  # Mild imbalance
            flip_y=0.05,  # 5% label noise
            random_state=self.random_state
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(25)])
        df['target'] = y
        
        issues_summary = {}
        
        # Issue 1: Missing values (MCAR pattern)
        missing_cols = ['feature_0', 'feature_5', 'feature_10']
        for col in missing_cols:
            mask = np.random.random(len(df)) < 0.08
            df.loc[mask, col] = np.nan
        issues_summary['missing_values'] = {'columns': missing_cols, 'rate': 0.08}
        
        # Issue 2: Outliers
        outlier_idx = np.random.choice(len(df), size=50, replace=False)
        df.loc[outlier_idx, 'feature_1'] = df['feature_1'].mean() + 8 * df['feature_1'].std()
        issues_summary['outliers'] = {'column': 'feature_1', 'count': 50}
        
        # Issue 3: Perfect correlation (leakage)
        df['leak_feature'] = df['feature_2'].copy()
        issues_summary['perfect_correlation'] = {'pair': ('leak_feature', 'feature_2')}
        
        # Issue 4: Duplicate feature
        df['dup_feature'] = df['feature_3'].copy()
        issues_summary['duplicate'] = {'pair': ('dup_feature', 'feature_3')}
        
        # Issue 5: High cardinality categorical
        df['high_card_cat'] = np.random.randint(0, 500, len(df)).astype(str)
        issues_summary['high_cardinality'] = {'column': 'high_card_cat', 'unique_values': 500}
        
        # Issue 6: Constant feature
        df['constant_feature'] = 42
        issues_summary['constant'] = {'column': 'constant_feature'}
        
        # Issue 7: Target leakage
        df['target_leak'] = df['target'] * 0.9 + np.random.randn(len(df)) * 0.1
        issues_summary['target_leakage'] = {'column': 'target_leak'}
        
        return BenchmarkDataset(
            name="benchmark_10_mixed_issues",
            data=df,
            target_column='target',
            known_issues={
                'issue_types': ['missing_values', 'outliers', 'perfect_correlation',
                               'duplicate_features', 'high_cardinality', 'constant_feature',
                               'target_leakage', 'class_imbalance', 'label_noise'],
                'details': issues_summary,
                'total_issue_count': 9
            },
            ground_truth={
                'missing_columns': missing_cols,
                'outlier_column': 'feature_1',
                'leaky_columns': ['leak_feature', 'target_leak'],
                'duplicate_columns': ['dup_feature'],
                'problematic_columns': ['high_card_cat', 'constant_feature'],
                'class_imbalance_ratio': 4.0,
                'label_noise_rate': 0.05
            },
            description="Complex dataset with multiple combined data quality issues: "
                       "missing values, outliers, correlations, duplicates, high cardinality, "
                       "constant features, target leakage, class imbalance, and label noise.",
            issue_type='mixed',
            difficulty='hard',
            expected_detection={
                'overall_issue_detection': 0.85,
                'individual_issue_detection': 0.80
            }
        )
    
    def _save_dataset(self, dataset: BenchmarkDataset, save_path: Path):
        """Save dataset and metadata to files"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_file = save_path / f"{dataset.name}.csv"
        dataset.data.to_csv(data_file, index=False)
        
        # Save metadata
        metadata = {
            'name': dataset.name,
            'target_column': dataset.target_column,
            'known_issues': dataset.known_issues,
            'ground_truth': dataset.ground_truth,
            'description': dataset.description,
            'issue_type': dataset.issue_type,
            'difficulty': dataset.difficulty,
            'expected_detection': dataset.expected_detection,
            'shape': list(dataset.data.shape),
            'columns': list(dataset.data.columns),
            'checksum': hashlib.md5(dataset.data.to_csv().encode()).hexdigest()
        }
        
        metadata_file = save_path / f"{dataset.name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """Get summary table of all benchmark datasets"""
        datasets = self.generate_all_benchmarks()
        
        summary_data = []
        for name, ds in datasets.items():
            summary_data.append({
                'Dataset': name,
                'Issue Type': ds.issue_type,
                'Difficulty': ds.difficulty,
                'Rows': ds.data.shape[0],
                'Columns': ds.data.shape[1],
                'Description': ds.description[:80] + '...'
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Generate and save all benchmark datasets"""
    generator = BenchmarkDatasetGenerator(random_state=42)
    
    # Generate all datasets
    save_path = Path(__file__).parent.parent.parent / 'benchmark_datasets'
    datasets = generator.generate_all_benchmarks(save_path=save_path)
    
    print("Generated Benchmark Datasets:")
    print("=" * 80)
    
    for name, dataset in datasets.items():
        print(f"\n{name}")
        print(f"  Type: {dataset.issue_type}")
        print(f"  Difficulty: {dataset.difficulty}")
        print(f"  Shape: {dataset.data.shape}")
        print(f"  Description: {dataset.description[:60]}...")
    
    print(f"\nDatasets saved to: {save_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(generator.get_dataset_summary().to_string(index=False))


if __name__ == "__main__":
    main()

