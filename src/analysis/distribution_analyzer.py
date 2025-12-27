"""
Data distribution analysis module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.stats import normaltest, jarque_bera, anderson
import warnings


class DistributionAnalyzer:
    """Analyzes data distributions and statistical properties"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.skewness_threshold = 1.0  # Threshold for high skewness
        self.outlier_methods = ['iqr', 'zscore', 'modified_zscore']
        
    def analyze(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive distribution analysis
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column if available
            
        Returns:
            Dictionary containing distribution analysis results
        """
        results = {
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'statistical_summary': {},
            'distribution_tests': {},
            'outliers': {},
            'skewness_analysis': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Classify columns by type
            results.update(self._classify_columns(df))
            
            # Statistical summary for numeric columns
            if results['numeric_columns']:
                results['statistical_summary'] = self._compute_statistical_summary(df, results['numeric_columns'])
                
                # Distribution tests
                results['distribution_tests'] = self._test_distributions(df, results['numeric_columns'])
                
                # Outlier detection
                results['outliers'] = self._detect_outliers(df, results['numeric_columns'])
                
                # Skewness analysis
                results['skewness_analysis'] = self._analyze_skewness(df, results['numeric_columns'])
                
            # Categorical analysis
            if results['categorical_columns']:
                results['categorical_analysis'] = self._analyze_categorical(df, results['categorical_columns'])
                
            # Target-specific analysis
            if target_column and target_column in df.columns:
                results['target_analysis'] = self._analyze_target_distribution(df, target_column)
                
            # Class balance analysis (if target is categorical)
            if target_column and target_column in results['categorical_columns']:
                results['class_balance'] = self._analyze_class_balance(df, target_column)
                
            # Generate issues and recommendations
            results['issues'] = self._identify_issues(results)
            results['recommendations'] = self._generate_recommendations(results, df)
            
            self.logger.info("Distribution analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in distribution analysis: {str(e)}")
            results['error'] = str(e)
            
        return results
        
    def _classify_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Classify columns by data type"""
        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        
        for column in df.columns:
            dtype = df[column].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_columns.append(column)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                datetime_columns.append(column)
            else:
                # Check if it's a categorical-like column
                unique_ratio = df[column].nunique() / len(df)
                if unique_ratio < 0.5 or df[column].nunique() < 50:  # Heuristic for categorical
                    categorical_columns.append(column)
                else:
                    # Could be text data or high-cardinality categorical
                    categorical_columns.append(column)
                    
        return {
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'datetime_columns': datetime_columns
        }
        
    def _compute_statistical_summary(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute statistical summary for numeric columns"""
        summary = {}
        
        for column in numeric_columns:
            try:
                series = df[column].dropna()
                if len(series) == 0:
                    continue
                    
                summary[column] = {
                    'count': len(series),
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'var': float(series.var()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    '25%': float(series.quantile(0.25)),
                    '50%': float(series.quantile(0.50)),
                    '75%': float(series.quantile(0.75)),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'range': float(series.max() - series.min()),
                    'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                    'cv': float(series.std() / series.mean()) if series.mean() != 0 else np.inf
                }
                
            except Exception as e:
                self.logger.warning(f"Error computing statistics for column {column}: {str(e)}")
                
        return summary
        
    def _test_distributions(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """Test distributions for normality and other properties"""
        distribution_tests = {}
        
        for column in numeric_columns:
            try:
                series = df[column].dropna()
                if len(series) < 8:  # Minimum sample size for tests
                    continue
                    
                tests = {}
                
                # Shapiro-Wilk test (for sample size < 5000)
                if len(series) <= 5000:
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(series)
                        tests['shapiro_wilk'] = {
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': shapiro_p > 0.05
                        }
                    except Exception:
                        pass
                        
                # D'Agostino's normality test
                try:
                    dagostino_stat, dagostino_p = normaltest(series)
                    tests['dagostino'] = {
                        'statistic': float(dagostino_stat),
                        'p_value': float(dagostino_p),
                        'is_normal': dagostino_p > 0.05
                    }
                except Exception:
                    pass
                    
                # Jarque-Bera test
                try:
                    jb_stat, jb_p = jarque_bera(series)
                    tests['jarque_bera'] = {
                        'statistic': float(jb_stat),
                        'p_value': float(jb_p),
                        'is_normal': jb_p > 0.05
                    }
                except Exception:
                    pass
                    
                # Anderson-Darling test
                try:
                    ad_result = anderson(series, dist='norm')
                    tests['anderson_darling'] = {
                        'statistic': float(ad_result.statistic),
                        'critical_values': ad_result.critical_values.tolist(),
                        'significance_levels': ad_result.significance_level.tolist()
                    }
                except Exception:
                    pass
                    
                distribution_tests[column] = tests
                
            except Exception as e:
                self.logger.warning(f"Error in distribution tests for column {column}: {str(e)}")
                
        return distribution_tests
        
    def _detect_outliers(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """Detect outliers using multiple methods"""
        outliers = {}
        
        for column in numeric_columns:
            try:
                series = df[column].dropna()
                if len(series) == 0:
                    continue
                    
                column_outliers = {}
                
                # IQR method
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
                column_outliers['iqr'] = {
                    'count': len(iqr_outliers),
                    'percentage': (len(iqr_outliers) / len(series)) * 100,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_values': iqr_outliers.tolist()[:50]  # Limit to first 50
                }
                
                # Z-score method
                z_scores = np.abs(stats.zscore(series))
                zscore_outliers = series[z_scores > 3]
                column_outliers['zscore'] = {
                    'count': len(zscore_outliers),
                    'percentage': (len(zscore_outliers) / len(series)) * 100,
                    'threshold': 3.0,
                    'outlier_values': zscore_outliers.tolist()[:50]
                }
                
                # Modified Z-score method (using median)
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros_like(series)
                modified_zscore_outliers = series[np.abs(modified_z_scores) > 3.5]
                
                column_outliers['modified_zscore'] = {
                    'count': len(modified_zscore_outliers),
                    'percentage': (len(modified_zscore_outliers) / len(series)) * 100,
                    'threshold': 3.5,
                    'outlier_values': modified_zscore_outliers.tolist()[:50]
                }
                
                outliers[column] = column_outliers
                
            except Exception as e:
                self.logger.warning(f"Error detecting outliers for column {column}: {str(e)}")
                
        return outliers
        
    def _analyze_skewness(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """Analyze skewness of numeric columns"""
        skewness_analysis = {
            'highly_skewed_columns': [],
            'skewness_values': {},
            'skewness_interpretation': {}
        }
        
        for column in numeric_columns:
            try:
                series = df[column].dropna()
                if len(series) == 0:
                    continue
                    
                skewness = series.skew()
                skewness_analysis['skewness_values'][column] = float(skewness)
                
                # Interpret skewness
                if abs(skewness) < 0.5:
                    interpretation = "Approximately symmetric"
                elif abs(skewness) < 1.0:
                    interpretation = "Moderately skewed"
                else:
                    interpretation = "Highly skewed"
                    skewness_analysis['highly_skewed_columns'].append(column)
                    
                if skewness > 0:
                    interpretation += " (right-tailed)"
                elif skewness < 0:
                    interpretation += " (left-tailed)"
                    
                skewness_analysis['skewness_interpretation'][column] = interpretation
                
            except Exception as e:
                self.logger.warning(f"Error analyzing skewness for column {column}: {str(e)}")
                
        return skewness_analysis
        
    def _analyze_categorical(self, df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Any]:
        """Analyze categorical columns"""
        categorical_analysis = {}
        
        for column in categorical_columns:
            try:
                series = df[column].dropna()
                if len(series) == 0:
                    continue
                    
                value_counts = series.value_counts()
                
                analysis = {
                    'unique_count': len(value_counts),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'cardinality_ratio': len(value_counts) / len(series) if len(series) > 0 else 0,
                    'top_10_values': value_counts.head(10).to_dict()
                }
                
                # Check for high cardinality
                if analysis['cardinality_ratio'] > 0.9:
                    analysis['high_cardinality'] = True
                    analysis['warning'] = "Very high cardinality - might be an identifier column"
                elif analysis['cardinality_ratio'] > 0.5:
                    analysis['high_cardinality'] = True
                    analysis['warning'] = "High cardinality - consider feature engineering"
                else:
                    analysis['high_cardinality'] = False
                    
                categorical_analysis[column] = analysis
                
            except Exception as e:
                self.logger.warning(f"Error analyzing categorical column {column}: {str(e)}")
                
        return categorical_analysis
        
    def _analyze_target_distribution(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable distribution"""
        target_analysis = {}
        
        try:
            target_series = df[target_column].dropna()
            
            if pd.api.types.is_numeric_dtype(target_series):
                # Numeric target (regression)
                target_analysis['type'] = 'numeric'
                target_analysis['statistics'] = {
                    'mean': float(target_series.mean()),
                    'median': float(target_series.median()),
                    'std': float(target_series.std()),
                    'min': float(target_series.min()),
                    'max': float(target_series.max()),
                    'skewness': float(target_series.skew()),
                    'kurtosis': float(target_series.kurtosis())
                }
                
                # Check for distribution issues
                if abs(target_analysis['statistics']['skewness']) > 1.0:
                    target_analysis['issues'] = target_analysis.get('issues', [])
                    target_analysis['issues'].append("Target variable is highly skewed")
                    
            else:
                # Categorical target (classification)
                target_analysis['type'] = 'categorical'
                value_counts = target_series.value_counts()
                
                target_analysis['class_distribution'] = value_counts.to_dict()
                target_analysis['class_count'] = len(value_counts)
                target_analysis['most_frequent_class'] = value_counts.index[0]
                target_analysis['class_proportions'] = (value_counts / len(target_series)).to_dict()
                
        except Exception as e:
            self.logger.warning(f"Error analyzing target distribution: {str(e)}")
            target_analysis['error'] = str(e)
            
        return target_analysis
        
    def _analyze_class_balance(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze class balance for classification targets"""
        balance_analysis = {}
        
        try:
            target_series = df[target_column].dropna()
            value_counts = target_series.value_counts()
            
            total_samples = len(target_series)
            class_proportions = value_counts / total_samples
            
            balance_analysis = {
                'total_samples': total_samples,
                'class_counts': value_counts.to_dict(),
                'class_proportions': class_proportions.to_dict(),
                'majority_class': value_counts.index[0],
                'minority_class': value_counts.index[-1],
                'majority_proportion': float(class_proportions.iloc[0]),
                'minority_proportion': float(class_proportions.iloc[-1]),
                'imbalance_ratio': float(value_counts.iloc[0] / value_counts.iloc[-1])
            }
            
            # Determine balance level
            if balance_analysis['imbalance_ratio'] < 1.5:
                balance_analysis['balance_level'] = 'Balanced'
            elif balance_analysis['imbalance_ratio'] < 4:
                balance_analysis['balance_level'] = 'Slightly imbalanced'
            elif balance_analysis['imbalance_ratio'] < 10:
                balance_analysis['balance_level'] = 'Moderately imbalanced'
            else:
                balance_analysis['balance_level'] = 'Highly imbalanced'
                
        except Exception as e:
            self.logger.warning(f"Error analyzing class balance: {str(e)}")
            balance_analysis['error'] = str(e)
            
        return balance_analysis
        
    def _identify_issues(self, results: Dict[str, Any]) -> List[str]:
        """Identify potential issues in the data distribution"""
        issues = []
        
        # Check for outliers
        outliers = results.get('outliers', {})
        for column, outlier_info in outliers.items():
            iqr_outliers = outlier_info.get('iqr', {}).get('percentage', 0)
            if iqr_outliers > 10:  # More than 10% outliers
                issues.append(f"High percentage of outliers in {column}: {iqr_outliers:.1f}%")
                
        # Check for skewness
        skewness_analysis = results.get('skewness_analysis', {})
        highly_skewed = skewness_analysis.get('highly_skewed_columns', [])
        if highly_skewed:
            issues.append(f"Highly skewed columns detected: {', '.join(highly_skewed)}")
            
        # Check for high cardinality categorical variables
        categorical_analysis = results.get('categorical_analysis', {})
        for column, cat_info in categorical_analysis.items():
            if cat_info.get('high_cardinality', False):
                issues.append(f"High cardinality in {column}: {cat_info['unique_count']} unique values")
                
        # Check class balance
        class_balance = results.get('class_balance', {})
        if class_balance:
            balance_level = class_balance.get('balance_level', '')
            if 'imbalanced' in balance_level.lower():
                ratio = class_balance.get('imbalance_ratio', 1)
                issues.append(f"Class imbalance detected: {balance_level} (ratio: {ratio:.2f}:1)")
                
        return issues
        
    def _generate_recommendations(self, results: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on distribution analysis"""
        recommendations = []
        
        # Outlier recommendations
        outliers = results.get('outliers', {})
        for column, outlier_info in outliers.items():
            iqr_percentage = outlier_info.get('iqr', {}).get('percentage', 0)
            if iqr_percentage > 5:
                recommendations.append(f"📊 {column}: {iqr_percentage:.1f}% outliers detected")
                if iqr_percentage > 15:
                    recommendations.append(f"  • Consider outlier removal or transformation for {column}")
                else:
                    recommendations.append(f"  • Consider robust scaling or outlier-resistant algorithms for {column}")
                    
        # Skewness recommendations
        highly_skewed = results.get('skewness_analysis', {}).get('highly_skewed_columns', [])
        if highly_skewed:
            recommendations.append("📈 Highly skewed distributions detected:")
            for column in highly_skewed:
                skewness = results['skewness_analysis']['skewness_values'].get(column, 0)
                recommendations.append(f"  • {column} (skewness: {skewness:.2f})")
            recommendations.append("  • Consider log transformation, Box-Cox, or Yeo-Johnson transformation")
            
        # Categorical recommendations
        categorical_analysis = results.get('categorical_analysis', {})
        high_cardinality_cols = [
            col for col, info in categorical_analysis.items() 
            if info.get('high_cardinality', False)
        ]
        if high_cardinality_cols:
            recommendations.append("🏷️ High cardinality categorical variables:")
            for col in high_cardinality_cols:
                unique_count = categorical_analysis[col]['unique_count']
                recommendations.append(f"  • {col}: {unique_count} unique values")
            recommendations.append("  • Consider target encoding, frequency encoding, or dimensionality reduction")
            
        # Class balance recommendations
        class_balance = results.get('class_balance', {})
        if class_balance and 'imbalanced' in class_balance.get('balance_level', '').lower():
            ratio = class_balance.get('imbalance_ratio', 1)
            recommendations.append(f"⚖️ Class imbalance detected (ratio: {ratio:.2f}:1)")
            
            if ratio > 10:
                recommendations.append("  • Consider SMOTE, ADASYN, or other oversampling techniques")
                recommendations.append("  • Use stratified sampling for train/validation splits")
                recommendations.append("  • Consider cost-sensitive learning algorithms")
            elif ratio > 4:
                recommendations.append("  • Consider class weighting in your model")
                recommendations.append("  • Use stratified sampling for train/validation splits")
            else:
                recommendations.append("  • Monitor model performance on minority class")
                
        # Normality recommendations
        distribution_tests = results.get('distribution_tests', {})
        non_normal_cols = []
        for col, tests in distribution_tests.items():
            for test_name, test_result in tests.items():
                if test_name in ['shapiro_wilk', 'dagostino', 'jarque_bera']:
                    if not test_result.get('is_normal', True):
                        non_normal_cols.append(col)
                        break
                        
        if non_normal_cols:
            recommendations.append("📊 Non-normal distributions detected:")
            for col in non_normal_cols[:5]:  # Show first 5
                recommendations.append(f"  • {col}")
            if len(non_normal_cols) > 5:
                recommendations.append(f"  • ... and {len(non_normal_cols) - 5} more columns")
            recommendations.append("  • Consider non-parametric methods or distribution transformations")
            recommendations.append("  • Tree-based algorithms handle non-normal distributions well")
            
        return recommendations