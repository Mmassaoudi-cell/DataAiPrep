"""
Data completeness analysis module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats


class CompletenessAnalyzer:
    """Analyzes data completeness and missing value patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive completeness analysis
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column if available
            
        Returns:
            Dictionary containing completeness analysis results
        """
        results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': df.size,
            'missing_by_column': {},
            'missing_patterns': {},
            'completeness_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Basic missing value analysis
            results.update(self._analyze_missing_values(df))
            
            # Missing value patterns
            results.update(self._analyze_missing_patterns(df))
            
            # Target-specific analysis if target column provided
            if target_column and target_column in df.columns:
                results.update(self._analyze_target_completeness(df, target_column))
                
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results, df)
            
            # Calculate overall completeness score
            results['completeness_score'] = self._calculate_completeness_score(results)
            
            self.logger.info(f"Completeness analysis completed. Score: {results['completeness_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in completeness analysis: {str(e)}")
            results['error'] = str(e)
            
        return results
        
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic missing value statistics"""
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        
        missing_by_column = missing_counts.to_dict()
        missing_percentages = (missing_counts / total_rows * 100).to_dict()
        
        # Overall statistics
        total_missing = missing_counts.sum()
        overall_missing_percentage = (total_missing / df.size * 100) if df.size > 0 else 0
        
        # Column categories
        complete_columns = [col for col, count in missing_counts.items() if count == 0]
        incomplete_columns = [col for col, count in missing_counts.items() if count > 0]
        
        # Most problematic columns
        high_missing_threshold = 50.0  # 50% missing
        high_missing_columns = [
            col for col, pct in missing_percentages.items() 
            if pct >= high_missing_threshold
        ]
        
        # Find column with most missing data
        most_missing_column = None
        if incomplete_columns:
            max_missing_col = missing_counts.idxmax()
            most_missing_column = {
                'column': max_missing_col,
                'count': missing_counts[max_missing_col],
                'percentage': missing_percentages[max_missing_col]
            }
        
        return {
            'missing_by_column': missing_by_column,
            'missing_percentages': missing_percentages,
            'total_missing_values': int(total_missing),
            'overall_missing_percentage': overall_missing_percentage,
            'complete_columns': complete_columns,
            'incomplete_columns': incomplete_columns,
            'complete_columns_count': len(complete_columns),
            'incomplete_columns_count': len(incomplete_columns),
            'high_missing_columns': high_missing_columns,
            'most_missing_column': most_missing_column
        }
        
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        patterns = {}
        
        try:
            # Create missing data matrix
            missing_matrix = df.isnull()
            
            # Pattern analysis for small datasets
            if len(df) <= 10000:  # Limit pattern analysis for performance
                # Find common missing patterns
                pattern_counts = missing_matrix.value_counts()
                
                # Get top 10 most common patterns
                top_patterns = pattern_counts.head(10)
                patterns['common_patterns'] = [
                    {
                        'pattern': list(pattern),
                        'count': count,
                        'percentage': (count / len(df) * 100)
                    }
                    for pattern, count in top_patterns.items()
                ]
            
            # Correlation between missing values
            if len(missing_matrix.columns) > 1:
                missing_corr = missing_matrix.corr()
                
                # Find highly correlated missing patterns
                high_corr_pairs = []
                for i, col1 in enumerate(missing_corr.columns):
                    for j, col2 in enumerate(missing_corr.columns):
                        if i < j:  # Avoid duplicates
                            corr_value = missing_corr.loc[col1, col2]
                            if abs(corr_value) > 0.5:  # Threshold for high correlation
                                high_corr_pairs.append({
                                    'column1': col1,
                                    'column2': col2,
                                    'correlation': corr_value
                                })
                
                patterns['missing_correlations'] = high_corr_pairs
                
            # Missing data by row
            missing_per_row = missing_matrix.sum(axis=1)
            patterns['rows_with_missing'] = (missing_per_row > 0).sum()
            patterns['rows_completely_missing'] = (missing_per_row == len(df.columns)).sum()
            patterns['max_missing_per_row'] = missing_per_row.max()
            patterns['avg_missing_per_row'] = missing_per_row.mean()
            
        except Exception as e:
            self.logger.warning(f"Error in pattern analysis: {str(e)}")
            patterns['error'] = str(e)
            
        return {'missing_patterns': patterns}
        
    def _analyze_target_completeness(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze completeness specifically for target variable"""
        target_analysis = {}
        
        try:
            target_missing = df[target_column].isnull().sum()
            target_missing_pct = (target_missing / len(df) * 100)
            
            target_analysis = {
                'target_column': target_column,
                'target_missing_count': int(target_missing),
                'target_missing_percentage': target_missing_pct,
                'target_complete_count': len(df) - target_missing
            }
            
            # Analyze missingness in features when target is missing
            if target_missing > 0:
                target_missing_mask = df[target_column].isnull()
                feature_cols = [col for col in df.columns if col != target_column]
                
                # Missing rates in features when target is missing
                feature_missing_when_target_missing = {}
                for col in feature_cols:
                    missing_when_target_missing = df.loc[target_missing_mask, col].isnull().sum()
                    if target_missing > 0:
                        pct = (missing_when_target_missing / target_missing * 100)
                        feature_missing_when_target_missing[col] = {
                            'count': int(missing_when_target_missing),
                            'percentage': pct
                        }
                
                target_analysis['feature_missing_when_target_missing'] = feature_missing_when_target_missing
                
        except Exception as e:
            self.logger.warning(f"Error in target completeness analysis: {str(e)}")
            target_analysis['error'] = str(e)
            
        return {'target_analysis': target_analysis}
        
    def _generate_recommendations(self, results: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on completeness analysis"""
        recommendations = []
        
        overall_missing = results.get('overall_missing_percentage', 0)
        high_missing_cols = results.get('high_missing_columns', [])
        incomplete_cols_count = results.get('incomplete_columns_count', 0)
        
        # Overall data quality recommendations
        if overall_missing < 5:
            recommendations.append("✓ Excellent data completeness! Dataset is ready for ML training.")
        elif overall_missing < 15:
            recommendations.append("⚠ Good data completeness with minor missing values.")
            recommendations.append("Consider simple imputation strategies (mean/median for numeric, mode for categorical).")
        elif overall_missing < 30:
            recommendations.append("⚠ Moderate missing data detected.")
            recommendations.append("Investigate missing data mechanisms (MCAR, MAR, MNAR).")
            recommendations.append("Consider advanced imputation methods (KNN, iterative imputation).")
        else:
            recommendations.append("🚨 High missing data detected! Data quality issues likely.")
            recommendations.append("Investigate data collection process and consider data source quality.")
            recommendations.append("Consider collecting more data or improving data pipeline.")
        
        # Column-specific recommendations
        if high_missing_cols:
            recommendations.append(f"🚨 {len(high_missing_cols)} columns have >50% missing data:")
            for col in high_missing_cols[:5]:  # Show first 5
                recommendations.append(f"  • {col}: Consider removing or investigating data source")
            if len(high_missing_cols) > 5:
                recommendations.append(f"  • ... and {len(high_missing_cols) - 5} more columns")
                
        # Pattern-based recommendations
        missing_patterns = results.get('missing_patterns', {})
        high_corr_missing = missing_patterns.get('missing_correlations', [])
        
        if high_corr_missing:
            recommendations.append("📊 Correlated missing patterns detected:")
            for pair in high_corr_missing[:3]:  # Show first 3
                recommendations.append(
                    f"  • {pair['column1']} & {pair['column2']} "
                    f"(correlation: {pair['correlation']:.2f})"
                )
            recommendations.append("Consider joint imputation for correlated missing patterns.")
            
        # Target-specific recommendations
        target_analysis = results.get('target_analysis', {})
        if target_analysis:
            target_missing_pct = target_analysis.get('target_missing_percentage', 0)
            if target_missing_pct > 10:
                recommendations.append(
                    f"🚨 Target variable has {target_missing_pct:.1f}% missing values!"
                )
                recommendations.append("Consider removing rows with missing target values for supervised learning.")
            elif target_missing_pct > 0:
                recommendations.append(
                    f"⚠ Target variable has {target_missing_pct:.1f}% missing values."
                )
                
        # Imputation strategy recommendations
        if incomplete_cols_count > 0:
            recommendations.append("\n📋 Suggested imputation strategies:")
            
            for col in df.columns:
                missing_pct = results['missing_percentages'].get(col, 0)
                if 0 < missing_pct <= 5:
                    if df[col].dtype in ['int64', 'float64']:
                        recommendations.append(f"  • {col}: Mean/median imputation (low missing: {missing_pct:.1f}%)")
                    else:
                        recommendations.append(f"  • {col}: Mode imputation (low missing: {missing_pct:.1f}%)")
                elif 5 < missing_pct <= 20:
                    recommendations.append(f"  • {col}: KNN or iterative imputation (moderate missing: {missing_pct:.1f}%)")
                elif 20 < missing_pct <= 50:
                    recommendations.append(f"  • {col}: Consider advanced imputation or feature engineering (high missing: {missing_pct:.1f}%)")
                elif missing_pct > 50:
                    recommendations.append(f"  • {col}: Consider removal (very high missing: {missing_pct:.1f}%)")
                    
        return recommendations
        
    def _calculate_completeness_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall completeness score (0-100)"""
        try:
            # Base score from overall completeness
            overall_missing_pct = results.get('overall_missing_percentage', 0)
            base_score = max(0, 100 - overall_missing_pct)
            
            # Penalty for high missing columns
            high_missing_cols = len(results.get('high_missing_columns', []))
            total_cols = results.get('total_columns', 1)
            high_missing_penalty = (high_missing_cols / total_cols) * 30  # Up to 30 point penalty
            
            # Penalty for target missing (if applicable)
            target_penalty = 0
            target_analysis = results.get('target_analysis', {})
            if target_analysis:
                target_missing_pct = target_analysis.get('target_missing_percentage', 0)
                target_penalty = min(target_missing_pct / 2, 20)  # Up to 20 point penalty
                
            # Bonus for good patterns
            bonus = 0
            incomplete_cols = results.get('incomplete_columns_count', 0)
            total_cols = results.get('total_columns', 1)
            if incomplete_cols / total_cols < 0.1:  # Less than 10% columns have missing data
                bonus = 5
                
            final_score = max(0, min(100, base_score - high_missing_penalty - target_penalty + bonus))
            return round(final_score, 2)
            
        except Exception as e:
            self.logger.warning(f"Error calculating completeness score: {str(e)}")
            return 0.0
            
    def detect_missing_mechanism(self, df: pd.DataFrame, column: str, 
                                test_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Attempt to detect missing data mechanism for a specific column
        
        Args:
            df: DataFrame to analyze
            column: Column to test for missing mechanism
            test_columns: Columns to test against for MAR detection
            
        Returns:
            Dictionary with mechanism detection results
        """
        result = {
            'column': column,
            'mechanism': 'Unknown',
            'confidence': 0.0,
            'test_results': {},
            'recommendation': ''
        }
        
        try:
            if column not in df.columns:
                result['error'] = f"Column {column} not found"
                return result
                
            missing_mask = df[column].isnull()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                result['mechanism'] = 'No missing data'
                result['confidence'] = 1.0
                return result
                
            # Test for MCAR (Missing Completely at Random)
            # Simple test: check if missing pattern is random across other variables
            if test_columns is None:
                test_columns = [col for col in df.columns if col != column and df[col].dtype in ['int64', 'float64']]
                
            mcar_tests = []
            for test_col in test_columns[:5]:  # Limit to 5 columns for performance
                try:
                    # Chi-square test for independence
                    if df[test_col].dtype in ['int64', 'float64']:
                        # Create bins for continuous variables
                        test_col_binned = pd.cut(df[test_col], bins=5, duplicates='drop')
                        contingency = pd.crosstab(missing_mask, test_col_binned)
                        
                        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                            chi2, p_value = stats.chi2_contingency(contingency)[:2]
                            mcar_tests.append({
                                'test_column': test_col,
                                'chi2_statistic': chi2,
                                'p_value': p_value,
                                'independent': p_value > 0.05
                            })
                except Exception:
                    continue
                    
            result['test_results']['mcar_tests'] = mcar_tests
            
            # Determine mechanism based on tests
            if mcar_tests:
                independent_tests = [test for test in mcar_tests if test['independent']]
                independence_ratio = len(independent_tests) / len(mcar_tests)
                
                if independence_ratio > 0.8:
                    result['mechanism'] = 'Likely MCAR'
                    result['confidence'] = independence_ratio
                    result['recommendation'] = 'Simple imputation methods should work well (mean, median, mode)'
                elif independence_ratio > 0.4:
                    result['mechanism'] = 'Possibly MAR'
                    result['confidence'] = 1 - independence_ratio
                    result['recommendation'] = 'Consider advanced imputation methods that use other variables'
                else:
                    result['mechanism'] = 'Likely MNAR'
                    result['confidence'] = 1 - independence_ratio
                    result['recommendation'] = 'Investigate domain knowledge and consider specialized handling'
            else:
                result['mechanism'] = 'Insufficient data for testing'
                result['recommendation'] = 'Collect more data or consult domain experts'
                
        except Exception as e:
            result['error'] = str(e)
            
        return result