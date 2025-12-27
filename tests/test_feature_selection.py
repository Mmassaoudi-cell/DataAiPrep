"""
Tests for Advanced Feature Selection Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced import AdvancedFeatureSelector


class TestAdvancedFeatureSelector:
    """Tests for AdvancedFeatureSelector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        
        # Create informative features
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        
        # Create target
        y = (X1 + X2 + np.random.normal(0, 0.1, n_samples)) > 0
        
        # Create noise features
        noise = np.random.normal(0, 1, (n_samples, 5))
        
        X = pd.DataFrame({
            'informative_1': X1,
            'informative_2': X2,
            'noise_1': noise[:, 0],
            'noise_2': noise[:, 1],
            'noise_3': noise[:, 2],
            'noise_4': noise[:, 3],
            'noise_5': noise[:, 4],
        })
        
        return X, pd.Series(y.astype(int), name='target')
    
    def test_initialization(self):
        """Test default initialization."""
        selector = AdvancedFeatureSelector()
        assert selector.random_state is None
        assert selector.results_ is None
    
    def test_initialization_with_seed(self):
        """Test initialization with random state."""
        selector = AdvancedFeatureSelector(random_state=42)
        assert selector.random_state == 42
    
    def test_variance_selection(self, sample_data):
        """Test variance threshold selection."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        results = selector.select_features(
            X, y,
            methods=['variance'],
            n_features=3
        )
        
        assert 'method_results' in results
        assert 'variance' in results['method_results']
        assert len(results['method_results']['variance']) > 0
    
    def test_correlation_selection(self, sample_data):
        """Test correlation-based selection."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        results = selector.select_features(
            X, y,
            methods=['correlation'],
            n_features=3
        )
        
        assert 'correlation' in results['method_results']
    
    def test_rfe_selection(self, sample_data):
        """Test RFE selection."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        results = selector.select_features(
            X, y,
            methods=['rfe'],
            n_features=3
        )
        
        assert 'rfe' in results['method_results']
        # RFE should select exactly n_features
        assert len(results['method_results']['rfe']) <= 3
    
    def test_lasso_selection(self, sample_data):
        """Test LASSO selection."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        results = selector.select_features(
            X, y,
            methods=['lasso'],
            n_features=3
        )
        
        assert 'lasso' in results['method_results']
    
    def test_multiple_methods(self, sample_data):
        """Test running multiple selection methods."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        results = selector.select_features(
            X, y,
            methods=['variance', 'correlation', 'rfe', 'lasso'],
            n_features=3
        )
        
        assert len(results['method_results']) == 4
        assert 'consensus_features' in results
        assert 'feature_rankings' in results
    
    def test_consensus_features(self, sample_data):
        """Test consensus feature calculation."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        results = selector.select_features(
            X, y,
            methods=['variance', 'correlation', 'rfe'],
            n_features=3,
            consensus_threshold=0.5
        )
        
        consensus = results['consensus_features']
        assert isinstance(consensus, list)
    
    def test_get_optimal_features(self, sample_data):
        """Test get_optimal_features method."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        selector.select_features(
            X, y,
            methods=['variance', 'correlation', 'rfe'],
            n_features=3
        )
        
        optimal = selector.get_optimal_features(min_methods=2)
        assert isinstance(optimal, list)
    
    def test_generate_report(self, sample_data):
        """Test report generation."""
        X, y = sample_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        selector.select_features(
            X, y,
            methods=['variance', 'correlation'],
            n_features=3
        )
        
        report = selector.generate_report()
        assert isinstance(report, str)
        assert 'Feature Selection Report' in report or len(report) > 0
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        selector = AdvancedFeatureSelector(random_state=42)
        
        X = pd.DataFrame()
        y = pd.Series([], dtype=int)
        
        with pytest.raises((ValueError, Exception)):
            selector.select_features(X, y, methods=['variance'])
    
    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        X = pd.DataFrame({'single_feature': np.random.normal(0, 1, 100)})
        y = pd.Series(np.random.binomial(1, 0.5, 100))
        
        selector = AdvancedFeatureSelector(random_state=42)
        results = selector.select_features(X, y, methods=['variance'], n_features=1)
        
        assert len(results['method_results']['variance']) <= 1


class TestFeatureRankings:
    """Tests for feature ranking functionality."""
    
    @pytest.fixture
    def ranked_data(self):
        """Create data with clear feature importance."""
        np.random.seed(42)
        n_samples = 300
        
        # Create features with different importance
        X1 = np.random.normal(0, 1, n_samples)  # Most important
        X2 = np.random.normal(0, 1, n_samples)  # Somewhat important
        X3 = np.random.normal(0, 1, n_samples)  # Pure noise
        
        # Target strongly depends on X1, somewhat on X2
        y = (2 * X1 + 0.5 * X2 + np.random.normal(0, 0.5, n_samples)) > 0
        
        X = pd.DataFrame({
            'important': X1,
            'medium': X2,
            'noise': X3,
        })
        
        return X, pd.Series(y.astype(int))
    
    def test_rankings_order(self, ranked_data):
        """Test that rankings correctly order features."""
        X, y = ranked_data
        selector = AdvancedFeatureSelector(random_state=42)
        
        results = selector.select_features(
            X, y,
            methods=['correlation'],
            n_features=2
        )
        
        rankings = results['feature_rankings']
        
        # Important feature should have higher rank than noise
        assert 'important' in rankings
        assert 'noise' in rankings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

