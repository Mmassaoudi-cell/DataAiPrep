"""
Feature quality assessment module for DataAiPrep
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy.stats import chi2_contingency, f_oneway, pearsonr
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif, 
    f_regression, f_classif, chi2
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings


class FeatureQualityAnalyzer:
    """Analyzes feature quality and relevance for ML training"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.low_variance_threshold = 0.01
        self.correlation_threshold = 0.8
        self.importance_threshold = 0.001
        
    def analyze(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive feature quality analysis
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column if available
            
        Returns:
            Dictionary containing feature quality analysis results
        """
        results = {
            'total_features': len(df.columns) - (1 if target_column else 0),
            'target_column': target_column,
            'feature_quality_scores': {},
            'low_variance_features': [],
            'redundant_features': [],
            'irrelevant_features': [],
            'high_cardinality_features': [],
            'feature_importance': {},
            'feature_stability': {},
            'data_type_analysis': {},
            'recommendations': [],
            'quality_score': 0.0
        }
        
        try:
            feature_columns = [col for col in df.columns if col != target_column]
            
            # 1. Basic feature analysis
            results.update(self._analyze_basic_features(df, feature_columns))
            
            # 2. Variance analysis
            results.update(self._analyze_variance(df, feature_columns))
            
            # 3. Correlation analysis
            results.update(self._analyze_correlations(df, feature_columns))
            
            # 4. Cardinality analysis
            results.update(self._analyze_cardinality(df, feature_columns))
            
            # 5. Feature importance analysis (if target provided)
            if target_column and target_column in df.columns:
                results.update(self._analyze_feature_importance(df, feature_columns, target_column))
                
            # 6. Feature stability analysis
            results.update(self._analyze_feature_stability(df, feature_columns))
            
            # 7. Data type consistency analysis
            results.update(self._analyze_data_types(df, feature_columns))
            
            # 8. Calculate individual feature quality scores
            results['feature_quality_scores'] = self._calculate_feature_scores(results, feature_columns)
            
            # 9. Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # 10. Calculate overall quality score
            results['quality_score'] = self._calculate_quality_score(results)
            
            self.logger.info(f"Feature quality analysis completed. Score: {results['quality_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in feature quality analysis: {str(e)}")
            results['error'] = str(e)
            
        return results
        
    def _analyze_basic_features(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze basic feature properties"""
        basic_analysis = {
            'numeric_features': [],
            'categorical_features': [],
            'datetime_features': [],
            'text_features': [],
            'feature_types': {}
        }
        
        try:
            for col in feature_columns:
                series = df[col]
                
                # Determine feature type
                if pd.api.types.is_numeric_dtype(series):
                    basic_analysis['numeric_features'].append(col)
                    feature_type = 'numeric'
                elif pd.api.types.is_datetime64_any_dtype(series):
                    basic_analysis['datetime_features'].append(col)
                    feature_type = 'datetime'
                elif series.dtype == 'object':
                    # Check if it's text or categorical
                    avg_length = series.dropna().astype(str).str.len().mean()
                    unique_ratio = series.nunique() / len(series)
                    
                    if avg_length > 50 or unique_ratio > 0.8:
                        basic_analysis['text_features'].append(col)
                        feature_type = 'text'
                    else:
                        basic_analysis['categorical_features'].append(col)
                        feature_type = 'categorical'
                else:
                    basic_analysis['categorical_features'].append(col)
                    feature_type = 'categorical'
                    
                basic_analysis['feature_types'][col] = feature_type
                
        except Exception as e:
            self.logger.warning(f"Error in basic feature analysis: {str(e)}")
            
        return basic_analysis
        
    def _analyze_variance(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze feature variance to identify low-variance features"""
        variance_analysis = {
            'low_variance_features': [],
            'feature_variances': {}
        }
        
        try:
            numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
            
            for col in numeric_features:
                try:
                    series = df[col].dropna()
                    if len(series) > 1:
                        variance = series.var()
                        variance_analysis['feature_variances'][col] = float(variance)
                        
                        # Check for low variance
                        if variance < self.low_variance_threshold:
                            variance_analysis['low_variance_features'].append({
                                'feature': col,
                                'variance': float(variance),
                                'reason': 'Low variance indicates limited information content'
                            })
                            
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error in variance analysis: {str(e)}")
            
        return variance_analysis
        
    def _analyze_correlations(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze feature correlations to identify redundant features"""
        correlation_analysis = {
            'redundant_features': [],
            'correlation_groups': []
        }
        
        try:
            numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_features) < 2:
                return correlation_analysis
                
            # Calculate correlation matrix
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_matrix = df[numeric_features].corr()
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i, col1 in enumerate(numeric_features):
                for j, col2 in enumerate(numeric_features):
                    if i < j:  # Avoid duplicates
                        corr_value = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_value) and abs(corr_value) >= self.correlation_threshold:
                            high_corr_pairs.append({
                                'feature1': col1,
                                'feature2': col2,
                                'correlation': float(corr_value),
                                'recommendation': f'Consider removing {col2}'
                            })
                            
            correlation_analysis['redundant_features'] = high_corr_pairs
            
            # Group highly correlated features
            correlation_groups = self._find_correlation_groups(corr_matrix, self.correlation_threshold)
            correlation_analysis['correlation_groups'] = correlation_groups
            
        except Exception as e:
            self.logger.warning(f"Error in correlation analysis: {str(e)}")
            
        return correlation_analysis
        
    def _analyze_cardinality(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze feature cardinality to identify problematic categorical features"""
        cardinality_analysis = {
            'high_cardinality_features': [],
            'cardinality_stats': {}
        }
        
        try:
            for col in feature_columns:
                series = df[col]
                unique_count = series.nunique()
                total_count = len(series)
                cardinality_ratio = unique_count / total_count if total_count > 0 else 0
                
                cardinality_analysis['cardinality_stats'][col] = {
                    'unique_count': int(unique_count),
                    'total_count': int(total_count),
                    'cardinality_ratio': float(cardinality_ratio)
                }
                
                # Check for high cardinality issues
                if series.dtype == 'object' and cardinality_ratio > 0.8:
                    cardinality_analysis['high_cardinality_features'].append({
                        'feature': col,
                        'unique_count': int(unique_count),
                        'cardinality_ratio': float(cardinality_ratio),
                        'issue': 'Very high cardinality - may be identifier column',
                        'recommendation': 'Consider removing or encoding differently'
                    })
                elif series.dtype == 'object' and unique_count > 100:
                    cardinality_analysis['high_cardinality_features'].append({
                        'feature': col,
                        'unique_count': int(unique_count),
                        'cardinality_ratio': float(cardinality_ratio),
                        'issue': 'High cardinality categorical feature',
                        'recommendation': 'Consider target encoding or dimensionality reduction'
                    })
                    
        except Exception as e:
            self.logger.warning(f"Error in cardinality analysis: {str(e)}")
            
        return cardinality_analysis
        
    def _analyze_feature_importance(self, df: pd.DataFrame, feature_columns: List[str], 
                                  target_column: str) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods"""
        importance_analysis = {
            'feature_importance': {},
            'irrelevant_features': [],
            'importance_rankings': {}
        }
        
        try:
            target_series = df[target_column].dropna()
            
            # Determine if target is numeric or categorical
            is_regression = pd.api.types.is_numeric_dtype(target_series)
            
            # Method 1: Statistical tests
            statistical_scores = self._calculate_statistical_importance(
                df, feature_columns, target_column, is_regression
            )
            importance_analysis['importance_rankings']['statistical'] = statistical_scores
            
            # Method 2: Mutual information
            mi_scores = self._calculate_mutual_information(
                df, feature_columns, target_column, is_regression
            )
            importance_analysis['importance_rankings']['mutual_information'] = mi_scores
            
            # Method 3: Random Forest importance (if enough data)
            if len(df) >= 100:  # Minimum samples for RF
                rf_scores = self._calculate_rf_importance(
                    df, feature_columns, target_column, is_regression
                )
                importance_analysis['importance_rankings']['random_forest'] = rf_scores
            
            # Combine importance scores
            combined_scores = self._combine_importance_scores(importance_analysis['importance_rankings'])
            importance_analysis['feature_importance'] = combined_scores
            
            # Identify irrelevant features
            for feature, score in combined_scores.items():
                if score < self.importance_threshold:
                    importance_analysis['irrelevant_features'].append({
                        'feature': feature,
                        'importance_score': float(score),
                        'reason': 'Low importance across multiple methods'
                    })
                    
        except Exception as e:
            self.logger.warning(f"Error in feature importance analysis: {str(e)}")
            
        return importance_analysis
        
    def _analyze_feature_stability(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze feature stability across data segments"""
        stability_analysis = {
            'feature_stability': {},
            'unstable_features': []
        }
        
        try:
            # Split data into segments for stability analysis
            n_segments = min(5, len(df) // 100)  # At least 100 samples per segment
            if n_segments < 2:
                return stability_analysis
                
            segment_size = len(df) // n_segments
            
            for col in feature_columns:
                try:
                    series = df[col]
                    
                    if pd.api.types.is_numeric_dtype(series):
                        # Calculate mean and std for each segment
                        segment_stats = []
                        for i in range(n_segments):
                            start_idx = i * segment_size
                            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(df)
                            segment_data = series.iloc[start_idx:end_idx].dropna()
                            
                            if len(segment_data) > 0:
                                segment_stats.append({
                                    'mean': segment_data.mean(),
                                    'std': segment_data.std()
                                })
                        
                        # Calculate stability metrics
                        if len(segment_stats) > 1:
                            means = [s['mean'] for s in segment_stats if not np.isnan(s['mean'])]
                            stds = [s['std'] for s in segment_stats if not np.isnan(s['std'])]
                            
                            if means:
                                mean_stability = np.std(means) / (np.mean(means) + 1e-8)  # CV of means
                                std_stability = np.std(stds) / (np.mean(stds) + 1e-8) if stds else 0
                                
                                stability_score = 1 / (1 + mean_stability + std_stability)
                                stability_analysis['feature_stability'][col] = float(stability_score)
                                
                                if stability_score < 0.7:  # Low stability threshold
                                    stability_analysis['unstable_features'].append({
                                        'feature': col,
                                        'stability_score': float(stability_score),
                                        'issue': 'Feature values vary significantly across data segments'
                                    })
                    else:
                        # For categorical features, check distribution stability
                        segment_distributions = []
                        for i in range(n_segments):
                            start_idx = i * segment_size
                            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(df)
                            segment_data = series.iloc[start_idx:end_idx].dropna()
                            
                            if len(segment_data) > 0:
                                dist = segment_data.value_counts(normalize=True)
                                segment_distributions.append(dist)
                        
                        # Calculate distribution stability using Jensen-Shannon divergence approximation
                        if len(segment_distributions) > 1:
                            # Simple stability measure: average pairwise correlation of distributions
                            correlations = []
                            for i in range(len(segment_distributions)):
                                for j in range(i + 1, len(segment_distributions)):
                                    dist1, dist2 = segment_distributions[i], segment_distributions[j]
                                    # Align distributions
                                    all_categories = dist1.index.union(dist2.index)
                                    aligned_dist1 = dist1.reindex(all_categories, fill_value=0)
                                    aligned_dist2 = dist2.reindex(all_categories, fill_value=0)
                                    
                                    corr, _ = pearsonr(aligned_dist1, aligned_dist2)
                                    if not np.isnan(corr):
                                        correlations.append(corr)
                            
                            if correlations:
                                stability_score = np.mean(correlations)
                                stability_analysis['feature_stability'][col] = float(stability_score)
                                
                                if stability_score < 0.6:
                                    stability_analysis['unstable_features'].append({
                                        'feature': col,
                                        'stability_score': float(stability_score),
                                        'issue': 'Feature distribution varies significantly across data segments'
                                    })
                                    
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error in stability analysis: {str(e)}")
            
        return stability_analysis
        
    def _analyze_data_types(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze data type consistency and appropriateness"""
        type_analysis = {
            'data_type_analysis': {},
            'type_issues': []
        }
        
        try:
            for col in feature_columns:
                series = df[col]
                
                analysis = {
                    'current_type': str(series.dtype),
                    'null_count': int(series.isnull().sum()),
                    'null_percentage': float(series.isnull().mean() * 100)
                }
                
                # Check for potential type issues
                if series.dtype == 'object':
                    # Check if it should be numeric
                    try:
                        numeric_series = pd.to_numeric(series, errors='coerce')
                        numeric_nulls = numeric_series.isnull().sum()
                        original_nulls = series.isnull().sum()
                        
                        if numeric_nulls - original_nulls < len(series) * 0.1:  # Less than 10% conversion errors
                            type_analysis['type_issues'].append({
                                'feature': col,
                                'issue': 'Should be numeric type',
                                'current_type': 'object',
                                'suggested_type': 'numeric',
                                'conversion_errors': int(numeric_nulls - original_nulls)
                            })
                            analysis['suggested_type'] = 'numeric'
                    except:
                        pass
                        
                    # Check if it should be datetime
                    try:
                        datetime_series = pd.to_datetime(series, errors='coerce')
                        datetime_nulls = datetime_series.isnull().sum()
                        original_nulls = series.isnull().sum()
                        
                        if datetime_nulls - original_nulls < len(series) * 0.1:
                            type_analysis['type_issues'].append({
                                'feature': col,
                                'issue': 'Should be datetime type',
                                'current_type': 'object',
                                'suggested_type': 'datetime',
                                'conversion_errors': int(datetime_nulls - original_nulls)
                            })
                            analysis['suggested_type'] = 'datetime'
                    except:
                        pass
                        
                # Check for mixed types in numeric columns
                elif pd.api.types.is_numeric_dtype(series):
                    # Check if it should be integer
                    if series.dtype == 'float64':
                        non_null_series = series.dropna()
                        if len(non_null_series) > 0 and (non_null_series % 1 == 0).all():
                            analysis['suggested_type'] = 'integer'
                            
                type_analysis['data_type_analysis'][col] = analysis
                
        except Exception as e:
            self.logger.warning(f"Error in data type analysis: {str(e)}")
            
        return type_analysis
        
    def _calculate_statistical_importance(self, df: pd.DataFrame, feature_columns: List[str], 
                                        target_column: str, is_regression: bool) -> Dict[str, float]:
        """Calculate feature importance using statistical tests"""
        scores = {}
        
        try:
            target_series = df[target_column].dropna()
            
            for feature in feature_columns:
                try:
                    feature_series = df[feature]
                    
                    # Align data
                    aligned_df = pd.DataFrame({
                        'feature': feature_series, 
                        'target': target_series
                    }).dropna()
                    
                    if len(aligned_df) < 10:  # Need minimum samples
                        scores[feature] = 0.0
                        continue
                        
                    aligned_feature = aligned_df['feature']
                    aligned_target = aligned_df['target']
                    
                    if pd.api.types.is_numeric_dtype(aligned_feature):
                        if is_regression:
                            # F-test for regression
                            f_stat, p_value = f_regression(
                                aligned_feature.values.reshape(-1, 1), 
                                aligned_target.values
                            )
                            score = f_stat[0] if len(f_stat) > 0 else 0.0
                        else:
                            # F-test for classification (ANOVA)
                            groups = [aligned_feature[aligned_target == cls].values 
                                    for cls in aligned_target.unique()]
                            groups = [g for g in groups if len(g) > 0]
                            
                            if len(groups) > 1:
                                f_stat, p_value = f_oneway(*groups)
                                score = f_stat if not np.isnan(f_stat) else 0.0
                            else:
                                score = 0.0
                    else:
                        # Categorical feature
                        if is_regression:
                            # ANOVA for categorical vs numeric
                            groups = [aligned_target[aligned_feature == cat].values 
                                    for cat in aligned_feature.unique()]
                            groups = [g for g in groups if len(g) > 0]
                            
                            if len(groups) > 1:
                                f_stat, p_value = f_oneway(*groups)
                                score = f_stat if not np.isnan(f_stat) else 0.0
                            else:
                                score = 0.0
                        else:
                            # Chi-square test for categorical vs categorical
                            try:
                                contingency_table = pd.crosstab(aligned_feature, aligned_target)
                                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                                score = chi2_stat
                            except:
                                score = 0.0
                    
                    # Normalize score
                    scores[feature] = min(score / 100.0, 1.0)  # Simple normalization
                    
                except Exception:
                    scores[feature] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"Error calculating statistical importance: {str(e)}")
            
        return scores
        
    def _calculate_mutual_information(self, df: pd.DataFrame, feature_columns: List[str], 
                                    target_column: str, is_regression: bool) -> Dict[str, float]:
        """Calculate mutual information scores"""
        scores = {}
        
        try:
            target_series = df[target_column].dropna()
            
            for feature in feature_columns[:50]:  # Limit for performance
                try:
                    feature_series = df[feature]
                    
                    # Align data
                    aligned_df = pd.DataFrame({
                        'feature': feature_series, 
                        'target': target_series
                    }).dropna()
                    
                    if len(aligned_df) < 10:
                        scores[feature] = 0.0
                        continue
                        
                    aligned_feature = aligned_df['feature']
                    aligned_target = aligned_df['target']
                    
                    if pd.api.types.is_numeric_dtype(aligned_feature):
                        if is_regression:
                            mi_score = mutual_info_regression(
                                aligned_feature.values.reshape(-1, 1), 
                                aligned_target.values
                            )[0]
                        else:
                            mi_score = mutual_info_classif(
                                aligned_feature.values.reshape(-1, 1), 
                                aligned_target.values
                            )[0]
                    else:
                        # Encode categorical feature
                        le = LabelEncoder()
                        encoded_feature = le.fit_transform(aligned_feature.astype(str))
                        
                        if is_regression:
                            mi_score = mutual_info_regression(
                                encoded_feature.reshape(-1, 1), 
                                aligned_target.values
                            )[0]
                        else:
                            mi_score = mutual_info_classif(
                                encoded_feature.reshape(-1, 1), 
                                aligned_target.values
                            )[0]
                    
                    scores[feature] = float(mi_score)
                    
                except Exception:
                    scores[feature] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"Error calculating mutual information: {str(e)}")
            
        return scores
        
    def _calculate_rf_importance(self, df: pd.DataFrame, feature_columns: List[str], 
                               target_column: str, is_regression: bool) -> Dict[str, float]:
        """Calculate Random Forest feature importance"""
        scores = {}
        
        try:
            # Prepare data
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Remove rows with missing target
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 50:  # Need minimum samples
                return scores
                
            # Handle missing values and categorical features
            X_processed = X.copy()
            
            # Fill missing values
            for col in X_processed.columns:
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                else:
                    X_processed[col] = X_processed[col].fillna('missing')
                    
            # Encode categorical features
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
            # Train Random Forest
            if is_regression:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            else:
                rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
                
            rf.fit(X_processed, y)
            
            # Get feature importance
            importance_scores = rf.feature_importances_
            
            for i, feature in enumerate(feature_columns):
                scores[feature] = float(importance_scores[i])
                
        except Exception as e:
            self.logger.warning(f"Error calculating RF importance: {str(e)}")
            
        return scores
        
    def _combine_importance_scores(self, importance_rankings: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine multiple importance ranking methods"""
        combined_scores = {}
        
        try:
            # Get all features
            all_features = set()
            for method_scores in importance_rankings.values():
                all_features.update(method_scores.keys())
            
            # Calculate combined score for each feature
            for feature in all_features:
                scores = []
                for method, method_scores in importance_rankings.items():
                    if feature in method_scores:
                        scores.append(method_scores[feature])
                
                if scores:
                    # Use mean of available scores
                    combined_scores[feature] = np.mean(scores)
                else:
                    combined_scores[feature] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"Error combining importance scores: {str(e)}")
            
        return combined_scores
        
    def _find_correlation_groups(self, corr_matrix: pd.DataFrame, threshold: float) -> List[List[str]]:
        """Find groups of highly correlated features"""
        groups = []
        
        try:
            features = corr_matrix.columns.tolist()
            remaining_features = set(features)
            
            while remaining_features:
                # Start new group with first remaining feature
                seed_feature = next(iter(remaining_features))
                current_group = [seed_feature]
                remaining_features.remove(seed_feature)
                
                # Find all features highly correlated with this group
                for feature in list(remaining_features):
                    max_corr_in_group = max(
                        abs(corr_matrix.loc[feature, group_member]) 
                        for group_member in current_group
                        if not pd.isna(corr_matrix.loc[feature, group_member])
                    )
                    
                    if max_corr_in_group >= threshold:
                        current_group.append(feature)
                        remaining_features.remove(feature)
                
                # Only add groups with more than one feature
                if len(current_group) > 1:
                    groups.append(current_group)
                    
        except Exception as e:
            self.logger.warning(f"Error finding correlation groups: {str(e)}")
            
        return groups
        
    def _calculate_feature_scores(self, results: Dict[str, Any], feature_columns: List[str]) -> Dict[str, float]:
        """Calculate individual feature quality scores"""
        feature_scores = {}
        
        try:
            for feature in feature_columns:
                score = 100.0  # Start with perfect score
                
                # Penalty for low variance
                if feature in [f['feature'] for f in results.get('low_variance_features', [])]:
                    score -= 30
                    
                # Penalty for being redundant
                if feature in [f['feature2'] for f in results.get('redundant_features', [])]:
                    score -= 25
                    
                # Penalty for high cardinality issues
                if feature in [f['feature'] for f in results.get('high_cardinality_features', [])]:
                    score -= 20
                    
                # Penalty for being irrelevant
                if feature in [f['feature'] for f in results.get('irrelevant_features', [])]:
                    score -= 40
                    
                # Penalty for instability
                if feature in [f['feature'] for f in results.get('unstable_features', [])]:
                    score -= 15
                    
                # Bonus for high importance
                importance = results.get('feature_importance', {}).get(feature, 0)
                if importance > 0.1:
                    score += 10
                elif importance > 0.05:
                    score += 5
                    
                feature_scores[feature] = max(0, min(100, score))
                
        except Exception as e:
            self.logger.warning(f"Error calculating feature scores: {str(e)}")
            
        return feature_scores
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on feature quality analysis"""
        recommendations = []
        
        # Low variance features
        low_var = results.get('low_variance_features', [])
        if low_var:
            recommendations.append(f"🔒 {len(low_var)} low-variance features detected!")
            recommendations.append("  • Consider removing features with very low variance")
            for feature in low_var[:3]:
                recommendations.append(f"    - {feature['feature']} (variance: {feature['variance']:.6f})")
                
        # Redundant features
        redundant = results.get('redundant_features', [])
        if redundant:
            recommendations.append(f"🔄 {len(redundant)} redundant feature pairs detected!")
            recommendations.append("  • Remove highly correlated features to reduce multicollinearity")
            for pair in redundant[:3]:
                recommendations.append(f"    - {pair['recommendation']} (corr: {pair['correlation']:.3f})")
                
        # High cardinality features
        high_card = results.get('high_cardinality_features', [])
        if high_card:
            recommendations.append(f"🏷️ {len(high_card)} high-cardinality features detected!")
            recommendations.append("  • Consider target encoding or dimensionality reduction")
            for feature in high_card[:3]:
                recommendations.append(f"    - {feature['feature']}: {feature['unique_count']} unique values")
                
        # Irrelevant features
        irrelevant = results.get('irrelevant_features', [])
        if irrelevant:
            recommendations.append(f"❌ {len(irrelevant)} irrelevant features detected!")
            recommendations.append("  • Remove features with low predictive power")
            for feature in irrelevant[:3]:
                recommendations.append(f"    - {feature['feature']} (importance: {feature['importance_score']:.4f})")
                
        # Unstable features
        unstable = results.get('unstable_features', [])
        if unstable:
            recommendations.append(f"📈 {len(unstable)} unstable features detected!")
            recommendations.append("  • Investigate features with inconsistent distributions")
            for feature in unstable[:3]:
                recommendations.append(f"    - {feature['feature']} (stability: {feature['stability_score']:.3f})")
                
        # Data type issues
        type_issues = results.get('data_type_analysis', {}).get('type_issues', [])
        if type_issues:
            recommendations.append(f"🔧 {len(type_issues)} data type issues detected!")
            recommendations.append("  • Fix data type inconsistencies for better performance")
            for issue in type_issues[:3]:
                recommendations.append(f"    - {issue['feature']}: {issue['issue']}")
                
        # General recommendations
        if not any([low_var, redundant, high_card, irrelevant, unstable, type_issues]):
            recommendations.append("✅ Feature quality looks good overall!")
            recommendations.append("  • No major feature quality issues detected")
        else:
            recommendations.append("\n📋 General Feature Engineering Recommendations:")
            recommendations.append("  • Perform feature selection based on importance scores")
            recommendations.append("  • Consider creating interaction features for important variables")
            recommendations.append("  • Apply appropriate scaling/normalization for numeric features")
            recommendations.append("  • Use proper encoding techniques for categorical features")
            
        return recommendations
        
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall feature quality score (0-100)"""
        try:
            base_score = 100.0
            
            # Penalties for various issues
            low_var_penalty = len(results.get('low_variance_features', [])) * 5
            redundant_penalty = len(results.get('redundant_features', [])) * 8
            high_card_penalty = len(results.get('high_cardinality_features', [])) * 6
            irrelevant_penalty = len(results.get('irrelevant_features', [])) * 10
            unstable_penalty = len(results.get('unstable_features', [])) * 4
            type_issues_penalty = len(results.get('data_type_analysis', {}).get('type_issues', [])) * 3
            
            total_penalty = (low_var_penalty + redundant_penalty + high_card_penalty + 
                           irrelevant_penalty + unstable_penalty + type_issues_penalty)
            
            # Bonus for having target and importance analysis
            if results.get('target_column'):
                base_score += 5
                
            final_score = max(0, min(100, base_score - total_penalty))
            return round(final_score, 2)
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {str(e)}")
            return 50.0