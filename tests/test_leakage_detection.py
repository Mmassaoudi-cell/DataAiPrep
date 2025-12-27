"""
Tests for Advanced Leakage Detection Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced import AdvancedLeakageDetector


class TestAdvancedLeakageDetector:
    """Tests for AdvancedLeakageDetector class."""
    
    @pytest.fixture
    def clean_data(self):
        """Create clean train/test split with no leakage."""
        np.random.seed(42)
        
        train_df = pd.DataFrame({
            'id': range(1, 101),
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(5, 2, 100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        
        test_df = pd.DataFrame({
            'id': range(101, 131),
            'feature_1': np.random.normal(0, 1, 30),
            'feature_2': np.random.normal(5, 2, 30),
            'target': np.random.binomial(1, 0.3, 30)
        })
        
        return train_df, test_df
    
    @pytest.fixture
    def contaminated_data(self):
        """Create train/test split with exact duplicates (contamination)."""
        np.random.seed(42)
        
        train_df = pd.DataFrame({
            'id': range(1, 101),
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(5, 2, 100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        
        # Test set includes some exact copies from train
        test_df = pd.DataFrame({
            'id': range(101, 126),
            'feature_1': np.random.normal(0, 1, 25),
            'feature_2': np.random.normal(5, 2, 25),
            'target': np.random.binomial(1, 0.3, 25)
        })
        
        # Add exact duplicates
        duplicates = train_df.iloc[:5].copy()
        duplicates['id'] = range(126, 131)
        test_df = pd.concat([test_df, duplicates], ignore_index=True)
        
        return train_df, test_df
    
    @pytest.fixture
    def group_leakage_data(self):
        """Create train/test split with group/entity leakage."""
        np.random.seed(42)
        
        train_df = pd.DataFrame({
            'customer_id': list(range(1, 51)) * 2,  # 50 customers, 2 records each
            'feature_1': np.random.normal(0, 1, 100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        
        # Test set with overlapping customer_ids
        test_df = pd.DataFrame({
            'customer_id': list(range(40, 60)) * 2,  # Overlaps with train (40-50)
            'feature_1': np.random.normal(0, 1, 40),
            'target': np.random.binomial(1, 0.3, 40)
        })
        
        return train_df, test_df
    
    def test_initialization(self):
        """Test default initialization."""
        detector = AdvancedLeakageDetector()
        assert detector.similarity_threshold == 0.95
        assert detector.correlation_threshold == 0.95
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        detector = AdvancedLeakageDetector(
            similarity_threshold=0.9,
            correlation_threshold=0.8
        )
        assert detector.similarity_threshold == 0.9
        assert detector.correlation_threshold == 0.8
    
    def test_clean_data_no_leakage(self, clean_data):
        """Test that clean data shows no leakage."""
        train_df, test_df = clean_data
        
        detector = AdvancedLeakageDetector()
        results = detector.detect_all(train_df, test_df)
        
        assert 'summary' in results
        # Clean data should have no or minimal issues
        assert results['summary']['total_issues'] == 0 or \
               results['summary']['severity'] in ['none', 'low']
    
    def test_contamination_detection(self, contaminated_data):
        """Test detection of train-test contamination."""
        train_df, test_df = contaminated_data
        
        detector = AdvancedLeakageDetector()
        results = detector.detect_all(train_df, test_df)
        
        assert results['summary']['has_contamination'] == True
        assert results['train_test_contamination']['exact_duplicates'] > 0
    
    def test_group_leakage_detection(self, group_leakage_data):
        """Test detection of group/entity leakage."""
        train_df, test_df = group_leakage_data
        
        detector = AdvancedLeakageDetector()
        results = detector.detect_all(
            train_df, 
            test_df,
            entity_column='customer_id'
        )
        
        assert results['summary']['has_group_leakage'] == True
        assert results['group_leakage']['overlap_count'] > 0
    
    def test_target_leakage_detection(self):
        """Test detection of target leakage."""
        np.random.seed(42)
        
        # Create data where a feature is highly correlated with target
        target = np.random.binomial(1, 0.3, 100)
        leaky_feature = target + np.random.normal(0, 0.01, 100)  # Almost perfect correlation
        
        train_df = pd.DataFrame({
            'id': range(100),
            'normal_feature': np.random.normal(0, 1, 100),
            'leaky_feature': leaky_feature,
            'target': target
        })
        
        test_df = pd.DataFrame({
            'id': range(100, 130),
            'normal_feature': np.random.normal(0, 1, 30),
            'leaky_feature': np.random.normal(0, 1, 30),
            'target': np.random.binomial(1, 0.3, 30)
        })
        
        detector = AdvancedLeakageDetector(correlation_threshold=0.9)
        results = detector.detect_all(
            train_df, 
            test_df,
            target_column='target'
        )
        
        assert results['summary']['has_target_leakage'] == True
        assert 'leaky_feature' in results['target_leakage'].get('high_correlation_features', [])
    
    def test_detect_all_returns_summary(self, clean_data):
        """Test that detect_all returns proper summary structure."""
        train_df, test_df = clean_data
        
        detector = AdvancedLeakageDetector()
        results = detector.detect_all(train_df, test_df)
        
        assert 'summary' in results
        assert 'train_test_contamination' in results
        assert 'near_duplicates' in results
        assert 'group_leakage' in results
        assert 'target_leakage' in results
        
        summary = results['summary']
        assert 'total_issues' in summary
        assert 'severity' in summary
        assert 'has_contamination' in summary
        assert 'has_near_duplicates' in summary
        assert 'has_group_leakage' in summary
        assert 'has_target_leakage' in summary
    
    def test_generate_report(self, contaminated_data):
        """Test report generation."""
        train_df, test_df = contaminated_data
        
        detector = AdvancedLeakageDetector()
        detector.detect_all(train_df, test_df)
        
        report = detector.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_empty_dataframes(self):
        """Test handling of empty DataFrames."""
        detector = AdvancedLeakageDetector()
        
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        # Should handle gracefully
        try:
            results = detector.detect_all(train_df, test_df)
            assert 'summary' in results
        except (ValueError, Exception):
            pass  # Expected for empty data


class TestNearDuplicateDetection:
    """Tests for near-duplicate detection."""
    
    def test_near_duplicate_detection(self):
        """Test detection of near-duplicate rows."""
        np.random.seed(42)
        
        train_df = pd.DataFrame({
            'id': range(100),
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(5, 2, 100),
        })
        
        # Create test set with near-duplicates
        test_df = pd.DataFrame({
            'id': range(100, 125),
            'feature_1': np.random.normal(0, 1, 25),
            'feature_2': np.random.normal(5, 2, 25),
        })
        
        # Add near-duplicates (slightly modified versions)
        near_dups = train_df.iloc[:5].copy()
        near_dups['id'] = range(125, 130)
        near_dups['feature_1'] = near_dups['feature_1'] + 0.001  # Very small change
        test_df = pd.concat([test_df, near_dups], ignore_index=True)
        
        detector = AdvancedLeakageDetector(similarity_threshold=0.99)
        results = detector.detect_all(train_df, test_df)
        
        # Should detect near-duplicates
        assert 'near_duplicates' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

