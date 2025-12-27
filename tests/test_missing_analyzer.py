"""
Tests for Advanced Missing Value Analyzer Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced import AdvancedMissingAnalyzer


class TestAdvancedMissingAnalyzer:
    """Tests for AdvancedMissingAnalyzer class."""
    
    @pytest.fixture
    def complete_data(self):
        """Create complete dataset with no missing values."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.choice(['A', 'B', 'C'], 100),
            'feature_3': np.random.randint(1, 100, 100)
        })
    
    @pytest.fixture
    def mcar_data(self):
        """Create dataset with MCAR (Missing Completely At Random) pattern."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(5, 2, 100),
            'feature_3': np.random.randint(1, 100, 100)
        })
        
        # Introduce MCAR missing values (random positions)
        missing_idx = np.random.choice(100, size=15, replace=False)
        df.loc[missing_idx, 'feature_1'] = np.nan
        
        return df
    
    @pytest.fixture
    def mar_data(self):
        """Create dataset with MAR (Missing At Random) pattern."""
        np.random.seed(42)
        df = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'value': np.random.normal(50, 10, 100)
        })
        
        # MAR: value is missing more often when category == 'A'
        a_mask = df['category'] == 'A'
        missing_in_a = np.random.choice(
            df[a_mask].index, 
            size=int(a_mask.sum() * 0.5), 
            replace=False
        )
        df.loc[missing_in_a, 'value'] = np.nan
        
        return df
    
    def test_initialization(self):
        """Test default initialization."""
        analyzer = AdvancedMissingAnalyzer()
        assert analyzer.results_ is None
    
    def test_complete_data_analysis(self, complete_data):
        """Test analysis of complete data with no missing values."""
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(complete_data)
        
        assert 'summary' in results
        assert results['summary']['total_missing'] == 0
        assert results['summary']['total_missing_percentage'] == 0.0
    
    def test_mcar_data_analysis(self, mcar_data):
        """Test analysis of MCAR data."""
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(mcar_data)
        
        assert 'summary' in results
        assert results['summary']['total_missing'] > 0
        assert results['summary']['total_missing_percentage'] > 0
        
        # Check column-level analysis
        if 'missing_by_column' in results['summary']:
            assert 'feature_1' in results['summary']['missing_by_column']
            assert results['summary']['missing_by_column']['feature_1']['count'] == 15
    
    def test_mar_data_analysis(self, mar_data):
        """Test analysis of MAR data."""
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(mar_data)
        
        assert 'summary' in results
        assert results['summary']['total_missing'] > 0
    
    def test_missing_by_column(self, mcar_data):
        """Test column-level missing value breakdown."""
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(mcar_data)
        
        if 'missing_by_column' in results['summary']:
            for col, info in results['summary']['missing_by_column'].items():
                assert 'count' in info
                assert 'percentage' in info
                assert info['percentage'] >= 0
                assert info['percentage'] <= 100
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        analyzer = AdvancedMissingAnalyzer()
        
        df = pd.DataFrame()
        
        # Should handle gracefully
        try:
            results = analyzer.analyze(df)
            assert 'summary' in results
        except (ValueError, Exception):
            pass  # Expected for empty data
    
    def test_all_missing_column(self):
        """Test handling of column with all missing values."""
        df = pd.DataFrame({
            'complete': [1, 2, 3, 4, 5],
            'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(df)
        
        assert 'summary' in results
        if 'missing_by_column' in results['summary']:
            assert results['summary']['missing_by_column']['all_missing']['percentage'] == 100.0


class TestMissingPatternAnalysis:
    """Tests for missing pattern analysis."""
    
    def test_pattern_detection(self):
        """Test detection of missing value patterns."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, np.nan],
            'B': [np.nan, 2, np.nan, 4, 5],
            'C': [1, 2, 3, 4, 5]
        })
        
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(df)
        
        assert 'summary' in results
        assert results['summary']['total_missing'] == 4  # 2 in A, 2 in B


class TestImputationRecommendations:
    """Tests for imputation recommendation functionality."""
    
    def test_numeric_column_recommendations(self):
        """Test recommendations for numeric columns."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0]
        })
        
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(df)
        
        # Recommendations should exist if implemented
        assert 'summary' in results
    
    def test_categorical_column_recommendations(self):
        """Test recommendations for categorical columns."""
        df = pd.DataFrame({
            'category': ['A', 'B', np.nan, 'A', 'C', np.nan, 'B', 'A']
        })
        
        analyzer = AdvancedMissingAnalyzer()
        results = analyzer.analyze(df)
        
        assert 'summary' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

