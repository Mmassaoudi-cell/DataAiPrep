"""
Advanced Missing Value Analysis Module

Features:
- MCAR/MAR/MNAR Classification using Little's MCAR test
- Missingness pattern mining
- Smart imputation recommendations
- Imputation impact simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MissingPattern:
    """Container for missing value pattern information"""
    pattern_id: int
    columns: List[str]
    frequency: int
    percentage: float
    classification: str  # MCAR, MAR, MNAR


@dataclass
class ImputationRecommendation:
    """Container for imputation recommendation"""
    column: str
    recommended_method: str
    alternative_methods: List[str]
    reasoning: str
    expected_impact: Dict[str, float]


class AdvancedMissingAnalyzer:
    """
    Advanced missing value analysis with MCAR/MAR/MNAR classification.
    
    Features:
    - Little's MCAR Test
    - Logistic regression-based MAR detection
    - Pattern mining for systematic missingness
    - Smart imputation strategy recommendations
    - Impact simulation for different imputation methods
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the analyzer.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.results_ = None
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive missing value analysis.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'summary': self._compute_summary(data),
            'patterns': self._mine_patterns(data),
            'classification': self._classify_missingness(data),
            'recommendations': self._generate_recommendations(data),
            'correlations': self._analyze_missingness_correlations(data)
        }
        
        self.results_ = results
        return results
    
    def _compute_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute missing value summary statistics"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        cols_with_missing = missing_counts[missing_counts > 0]
        
        return {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'columns_with_missing': len(cols_with_missing),
            'total_missing_cells': int(missing_counts.sum()),
            'total_missing_percentage': float(data.isnull().sum().sum() / data.size * 100),
            'missing_by_column': {
                col: {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col])
                }
                for col in cols_with_missing.index
            }
        }
    
    def _mine_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Mine systematic missing value patterns"""
        # Create binary missing indicator matrix
        missing_matrix = data.isnull().astype(int)
        
        # Find unique patterns
        pattern_counts = missing_matrix.groupby(list(missing_matrix.columns)).size()
        
        patterns = []
        for i, (pattern, count) in enumerate(pattern_counts.items()):
            if isinstance(pattern, tuple):
                missing_cols = [col for col, val in zip(missing_matrix.columns, pattern) if val == 1]
            else:
                missing_cols = [missing_matrix.columns[0]] if pattern == 1 else []
            
            if missing_cols:  # Only include patterns with missing values
                patterns.append({
                    'pattern_id': i,
                    'columns_missing': missing_cols,
                    'frequency': int(count),
                    'percentage': float(count / len(data) * 100)
                })
        
        # Sort by frequency
        patterns.sort(key=lambda x: x['frequency'], reverse=True)
        return patterns[:20]  # Top 20 patterns
    
    def _classify_missingness(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Classify missingness mechanism for each column.
        
        Uses:
        - Little's MCAR test (simplified version)
        - Logistic regression for MAR detection
        - Heuristics for MNAR detection
        """
        classifications = {}
        
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        for col in missing_cols:
            classification = self._classify_column(data, col)
            classifications[col] = classification
            
        return classifications
    
    def _classify_column(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Classify missingness mechanism for a single column"""
        missing_mask = data[column].isnull()
        
        if missing_mask.sum() == 0:
            return {'mechanism': 'complete', 'confidence': 1.0}
        
        # Get other columns for testing
        other_cols = [c for c in data.columns if c != column and data[c].dtype in ['int64', 'float64']]
        
        if not other_cols:
            return {
                'mechanism': 'MCAR',
                'confidence': 0.5,
                'note': 'Insufficient numeric columns for testing'
            }
        
        # Test 1: Little's MCAR test approximation
        # Compare means of other variables between missing/non-missing groups
        mcar_test_results = []
        for other_col in other_cols[:10]:  # Limit to 10 columns
            try:
                group_missing = data.loc[missing_mask, other_col].dropna()
                group_complete = data.loc[~missing_mask, other_col].dropna()
                
                if len(group_missing) > 5 and len(group_complete) > 5:
                    stat, pvalue = stats.ttest_ind(group_missing, group_complete)
                    mcar_test_results.append(pvalue)
            except Exception:
                continue
        
        # Test 2: Logistic regression for MAR
        mar_predictability = 0.5
        try:
            # Prepare features (impute missing with mean for this test)
            X = data[other_cols].copy()
            for c in X.columns:
                X[c] = X[c].fillna(X[c].mean())
            
            y = missing_mask.astype(int)
            
            if len(y.unique()) > 1 and len(y) > 20:
                model = LogisticRegression(max_iter=500, solver='lbfgs')
                model.fit(X, y)
                mar_predictability = model.score(X, y)
        except Exception:
            pass
        
        # Classification logic
        if len(mcar_test_results) > 0:
            avg_pvalue = np.mean(mcar_test_results)
            significant_tests = sum(p < self.alpha for p in mcar_test_results) / len(mcar_test_results)
        else:
            avg_pvalue = 1.0
            significant_tests = 0
        
        # Determine mechanism
        if mar_predictability > 0.7:
            mechanism = 'MAR'
            confidence = min(mar_predictability, 0.95)
            reasoning = f"Missingness is predictable from other variables (accuracy: {mar_predictability:.2%})"
        elif significant_tests < 0.2 and avg_pvalue > self.alpha:
            mechanism = 'MCAR'
            confidence = 1 - significant_tests
            reasoning = f"No significant relationship between missingness and other variables"
        elif significant_tests > 0.5:
            mechanism = 'MAR'
            confidence = significant_tests
            reasoning = f"{significant_tests:.0%} of variables show significant relationship with missingness"
        else:
            mechanism = 'MNAR'
            confidence = 0.6
            reasoning = "Unable to explain missingness from observed data; may depend on unobserved values"
        
        return {
            'mechanism': mechanism,
            'confidence': float(confidence),
            'reasoning': reasoning,
            'mar_predictability': float(mar_predictability),
            'mcar_avg_pvalue': float(avg_pvalue) if mcar_test_results else None
        }
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[ImputationRecommendation]:
        """Generate smart imputation recommendations for each column"""
        recommendations = []
        
        for col in data.columns[data.isnull().any()]:
            rec = self._recommend_imputation(data, col)
            recommendations.append(rec)
            
        return recommendations
    
    def _recommend_imputation(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Generate imputation recommendation for a single column"""
        series = data[column]
        missing_pct = series.isnull().sum() / len(series) * 100
        
        # Determine data type
        if series.dtype in ['int64', 'float64']:
            dtype = 'numeric'
        elif series.dtype == 'object' or series.dtype.name == 'category':
            dtype = 'categorical'
        else:
            dtype = 'other'
        
        # Get classification if available
        classification = self.results_['classification'].get(column, {}) if self.results_ else {}
        mechanism = classification.get('mechanism', 'unknown')
        
        # Generate recommendation based on mechanism and characteristics
        if dtype == 'numeric':
            skewness = abs(series.dropna().skew()) if len(series.dropna()) > 3 else 0
            
            if missing_pct > 50:
                method = 'drop_column'
                alternatives = ['indicator_variable', 'multiple_imputation']
                reasoning = f"High missing rate ({missing_pct:.1f}%) - consider dropping or creating indicator"
            elif mechanism == 'MCAR':
                if skewness > 1:
                    method = 'median'
                    alternatives = ['mean', 'knn', 'iterative']
                    reasoning = "MCAR pattern with skewed distribution - median preserves distribution"
                else:
                    method = 'mean'
                    alternatives = ['median', 'knn', 'iterative']
                    reasoning = "MCAR pattern with symmetric distribution - mean is unbiased"
            elif mechanism == 'MAR':
                method = 'knn'
                alternatives = ['iterative', 'regression', 'random_forest']
                reasoning = "MAR pattern - use model-based imputation leveraging observed relationships"
            else:  # MNAR or unknown
                method = 'multiple_imputation'
                alternatives = ['iterative', 'indicator_plus_imputation']
                reasoning = "Potential MNAR - multiple imputation accounts for uncertainty"
        else:  # categorical
            if missing_pct > 50:
                method = 'drop_column'
                alternatives = ['new_category', 'mode']
                reasoning = f"High missing rate ({missing_pct:.1f}%) for categorical"
            else:
                method = 'mode'
                alternatives = ['new_category', 'knn']
                reasoning = "Categorical column - mode imputation or create 'Missing' category"
        
        return {
            'column': column,
            'dtype': dtype,
            'missing_percentage': missing_pct,
            'mechanism': mechanism,
            'recommended_method': method,
            'alternative_methods': alternatives,
            'reasoning': reasoning
        }
    
    def _analyze_missingness_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between missing value indicators"""
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if len(missing_cols) < 2:
            return {'note': 'Insufficient columns with missing values for correlation analysis'}
        
        # Create missing indicator matrix
        missing_indicators = data[missing_cols].isnull().astype(int)
        
        # Compute correlation matrix
        corr_matrix = missing_indicators.corr()
        
        # Find highly correlated missing patterns
        high_correlations = []
        for i, col1 in enumerate(missing_cols):
            for j, col2 in enumerate(missing_cols):
                if i < j and abs(corr_matrix.loc[col1, col2]) > 0.5:
                    high_correlations.append({
                        'column_1': col1,
                        'column_2': col2,
                        'correlation': float(corr_matrix.loc[col1, col2])
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': sorted(high_correlations, 
                                       key=lambda x: abs(x['correlation']), 
                                       reverse=True)[:10]
        }
    
    def simulate_imputation_impact(
        self,
        data: pd.DataFrame,
        column: str,
        methods: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Simulate the impact of different imputation methods.
        
        Args:
            data: DataFrame with missing values
            column: Column to impute
            methods: List of methods to test
            
        Returns:
            Dictionary with distribution statistics for each method
        """
        if methods is None:
            methods = ['mean', 'median', 'mode', 'knn']
        
        results = {}
        original_stats = {
            'mean': float(data[column].mean()),
            'std': float(data[column].std()),
            'median': float(data[column].median()),
            'skewness': float(data[column].skew()),
            'missing_count': int(data[column].isnull().sum())
        }
        results['original'] = original_stats
        
        for method in methods:
            try:
                imputed = self._apply_imputation(data, column, method)
                results[method] = {
                    'mean': float(imputed[column].mean()),
                    'std': float(imputed[column].std()),
                    'median': float(imputed[column].median()),
                    'skewness': float(imputed[column].skew()),
                    'mean_change': float(imputed[column].mean() - original_stats['mean']),
                    'std_change': float(imputed[column].std() - original_stats['std'])
                }
            except Exception as e:
                results[method] = {'error': str(e)}
        
        return results
    
    def _apply_imputation(
        self,
        data: pd.DataFrame,
        column: str,
        method: str
    ) -> pd.DataFrame:
        """Apply a specific imputation method"""
        df = data.copy()
        
        if method == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif method == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif method == 'mode':
            df[column] = df[column].fillna(df[column].mode().iloc[0])
        elif method == 'knn':
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if column in numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        elif method == 'iterative':
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if column in numeric_cols:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df

