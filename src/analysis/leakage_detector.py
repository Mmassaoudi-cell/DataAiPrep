"""
Data leakage detection module for DataAiPrep
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings


class LeakageDetector:
    """Detects various types of data leakage that could cause overfitting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_threshold = 0.95  # High correlation threshold
        self.duplicate_threshold = 0.9     # Duplicate similarity threshold
        self.leakage_threshold = 0.8       # Information leakage threshold
        
    def analyze(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive leakage detection analysis
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column if available
            
        Returns:
            Dictionary containing leakage analysis results
        """
        results = {
            'target_column': target_column,
            'total_features': len(df.columns),
            'leakage_issues': [],
            'suspicious_features': [],
            'duplicate_features': [],
            'perfect_correlations': [],
            'temporal_leakage': [],
            'target_leakage': {},
            'recommendations': [],
            'leakage_score': 0.0
        }
        
        try:
            # 1. Perfect correlation detection
            results.update(self._detect_perfect_correlations(df))
            
            # 2. Duplicate feature detection
            results.update(self._detect_duplicate_features(df))
            
            # 3. Target leakage detection (if target provided)
            if target_column and target_column in df.columns:
                results.update(self._detect_target_leakage(df, target_column))
                
            # 4. Temporal leakage detection
            results.update(self._detect_temporal_leakage(df))
            
            # 5. Feature-target information leakage
            if target_column and target_column in df.columns:
                results.update(self._detect_information_leakage(df, target_column))
                
            # 6. Statistical leakage patterns
            results.update(self._detect_statistical_leakage(df, target_column))
            
            # 7. Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # 8. Calculate leakage score
            results['leakage_score'] = self._calculate_leakage_score(results)
            
            self.logger.info(f"Leakage detection completed. Score: {results['leakage_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in leakage detection: {str(e)}")
            results['error'] = str(e)
            
        return results
        
    def _detect_perfect_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect features with perfect or near-perfect correlations"""
        perfect_correlations = []
        
        try:
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'perfect_correlations': perfect_correlations}
                
            # Calculate correlation matrix
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_matrix = df[numeric_cols].corr()
            
            # Find high correlations (excluding diagonal)
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicates and diagonal
                        corr_value = corr_matrix.loc[col1, col2]
                        
                        if not pd.isna(corr_value) and abs(corr_value) >= self.correlation_threshold:
                            perfect_correlations.append({
                                'feature1': col1,
                                'feature2': col2,
                                'correlation': float(corr_value),
                                'type': 'Perfect positive' if corr_value > 0 else 'Perfect negative',
                                'severity': 'Critical' if abs(corr_value) >= 0.99 else 'High'
                            })
                            
        except Exception as e:
            self.logger.warning(f"Error detecting perfect correlations: {str(e)}")
            
        return {'perfect_correlations': perfect_correlations}
        
    def _detect_duplicate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate or near-duplicate features"""
        duplicate_features = []
        
        try:
            columns = df.columns.tolist()
            
            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    if i < j:  # Avoid duplicates
                        try:
                            # For numeric columns, check correlation
                            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                                # Skip if either column has too many NaNs
                                if df[col1].isna().sum() > len(df) * 0.5 or df[col2].isna().sum() > len(df) * 0.5:
                                    continue
                                    
                                corr, _ = pearsonr(df[col1].fillna(0), df[col2].fillna(0))
                                if not np.isnan(corr) and abs(corr) >= self.duplicate_threshold:
                                    duplicate_features.append({
                                        'feature1': col1,
                                        'feature2': col2,
                                        'similarity': float(abs(corr)),
                                        'type': 'Numeric correlation',
                                        'recommendation': f'Consider removing {col2}'
                                    })
                                    
                            # For categorical columns, check value similarity
                            elif df[col1].dtype == 'object' and df[col2].dtype == 'object':
                                # Compare unique values
                                unique1 = set(df[col1].dropna().astype(str))
                                unique2 = set(df[col2].dropna().astype(str))
                                
                                if unique1 and unique2:
                                    jaccard = len(unique1.intersection(unique2)) / len(unique1.union(unique2))
                                    if jaccard >= self.duplicate_threshold:
                                        duplicate_features.append({
                                            'feature1': col1,
                                            'feature2': col2,
                                            'similarity': float(jaccard),
                                            'type': 'Categorical overlap',
                                            'recommendation': f'Consider removing {col2}'
                                        })
                                        
                        except Exception:
                            continue
                            
        except Exception as e:
            self.logger.warning(f"Error detecting duplicate features: {str(e)}")
            
        return {'duplicate_features': duplicate_features}
        
    def _detect_target_leakage(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect features that leak information about the target"""
        target_leakage = {
            'perfect_predictors': [],
            'high_information_features': [],
            'suspicious_names': [],
            'target_statistics': {}
        }
        
        try:
            target_series = df[target_column]
            feature_columns = [col for col in df.columns if col != target_column]
            
            # Target statistics
            if pd.api.types.is_numeric_dtype(target_series):
                target_leakage['target_statistics'] = {
                    'type': 'numeric',
                    'mean': float(target_series.mean()),
                    'std': float(target_series.std()),
                    'unique_values': int(target_series.nunique())
                }
            else:
                target_leakage['target_statistics'] = {
                    'type': 'categorical',
                    'unique_values': int(target_series.nunique()),
                    'mode': target_series.mode().iloc[0] if len(target_series.mode()) > 0 else None,
                    'class_distribution': target_series.value_counts().to_dict()
                }
            
            # Check each feature for target leakage
            for feature in feature_columns:
                try:
                    # Perfect predictor detection
                    if self._is_perfect_predictor(df[feature], target_series):
                        target_leakage['perfect_predictors'].append({
                            'feature': feature,
                            'type': 'Perfect predictor',
                            'severity': 'Critical'
                        })
                        
                    # High information content
                    info_score = self._calculate_information_score(df[feature], target_series)
                    if info_score > self.leakage_threshold:
                        target_leakage['high_information_features'].append({
                            'feature': feature,
                            'information_score': float(info_score),
                            'severity': 'High' if info_score > 0.9 else 'Medium'
                        })
                        
                    # Suspicious feature names
                    if self._has_suspicious_name(feature, target_column):
                        target_leakage['suspicious_names'].append({
                            'feature': feature,
                            'reason': 'Feature name suggests target information',
                            'severity': 'Medium'
                        })
                        
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error detecting target leakage: {str(e)}")
            
        return {'target_leakage': target_leakage}
        
    def _detect_temporal_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential temporal leakage patterns"""
        temporal_leakage = []
        
        try:
            # Look for date/time columns
            datetime_columns = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_columns.append(col)
                elif df[col].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(df[col].dropna().head(100))
                        datetime_columns.append(col)
                    except:
                        pass
                        
            # Check for suspicious temporal patterns
            for col in datetime_columns:
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        date_series = pd.to_datetime(df[col], errors='coerce')
                    else:
                        date_series = df[col]
                        
                    # Check for future dates
                    current_date = pd.Timestamp.now()
                    future_dates = date_series > current_date
                    
                    if future_dates.any():
                        temporal_leakage.append({
                            'feature': col,
                            'issue': 'Future dates detected',
                            'count': int(future_dates.sum()),
                            'percentage': float(future_dates.mean() * 100),
                            'severity': 'Critical'
                        })
                        
                    # Check for unrealistic date ranges
                    if date_series.min() < pd.Timestamp('1900-01-01'):
                        temporal_leakage.append({
                            'feature': col,
                            'issue': 'Unrealistic historical dates',
                            'min_date': str(date_series.min()),
                            'severity': 'Medium'
                        })
                        
                except Exception:
                    continue
                    
            # Look for features with suspicious temporal names
            temporal_keywords = ['created', 'updated', 'modified', 'timestamp', 'date', 'time', 
                               'processed', 'completed', 'finished', 'ended', 'closed']
            
            for col in df.columns:
                col_lower = col.lower()
                for keyword in temporal_keywords:
                    if keyword in col_lower and col not in datetime_columns:
                        temporal_leakage.append({
                            'feature': col,
                            'issue': 'Suspicious temporal feature name',
                            'keyword': keyword,
                            'severity': 'Low'
                        })
                        break
                        
        except Exception as e:
            self.logger.warning(f"Error detecting temporal leakage: {str(e)}")
            
        return {'temporal_leakage': temporal_leakage}
        
    def _detect_information_leakage(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect information leakage using mutual information"""
        information_leakage = {
            'high_mutual_info_features': [],
            'feature_importance_leakage': []
        }
        
        try:
            target_series = df[target_column].dropna()
            feature_columns = [col for col in df.columns if col != target_column]
            
            # Calculate mutual information for each feature
            for feature in feature_columns[:50]:  # Limit to first 50 features for performance
                try:
                    feature_series = df[feature]
                    
                    # Align indices (remove rows where either target or feature is NaN)
                    valid_idx = target_series.index.intersection(feature_series.dropna().index)
                    if len(valid_idx) < 10:  # Need minimum samples
                        continue
                        
                    aligned_target = target_series.loc[valid_idx]
                    aligned_feature = feature_series.loc[valid_idx]
                    
                    # Calculate mutual information
                    if pd.api.types.is_numeric_dtype(aligned_feature):
                        if pd.api.types.is_numeric_dtype(aligned_target):
                            # Numeric feature, numeric target
                            mi_score = mutual_info_regression(
                                aligned_feature.values.reshape(-1, 1), 
                                aligned_target.values
                            )[0]
                        else:
                            # Numeric feature, categorical target
                            mi_score = mutual_info_classif(
                                aligned_feature.values.reshape(-1, 1), 
                                aligned_target.values
                            )[0]
                    else:
                        # Categorical feature - encode first
                        le = LabelEncoder()
                        try:
                            encoded_feature = le.fit_transform(aligned_feature.astype(str))
                            if pd.api.types.is_numeric_dtype(aligned_target):
                                mi_score = mutual_info_regression(
                                    encoded_feature.reshape(-1, 1), 
                                    aligned_target.values
                                )[0]
                            else:
                                mi_score = mutual_info_classif(
                                    encoded_feature.reshape(-1, 1), 
                                    aligned_target.values
                                )[0]
                        except:
                            continue
                    
                    # Normalize mutual information score
                    if mi_score > 0:
                        # Calculate maximum possible MI (entropy of target)
                        if pd.api.types.is_numeric_dtype(aligned_target):
                            # For continuous targets, use a heuristic normalization
                            normalized_mi = min(mi_score / 2.0, 1.0)  # Rough normalization
                        else:
                            # For categorical targets, normalize by target entropy
                            target_entropy = -np.sum(aligned_target.value_counts(normalize=True) * 
                                                   np.log2(aligned_target.value_counts(normalize=True)))
                            normalized_mi = mi_score / target_entropy if target_entropy > 0 else 0
                        
                        if normalized_mi > 0.6:  # High mutual information
                            information_leakage['high_mutual_info_features'].append({
                                'feature': feature,
                                'mutual_information': float(mi_score),
                                'normalized_mi': float(normalized_mi),
                                'severity': 'Critical' if normalized_mi > 0.8 else 'High'
                            })
                            
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error detecting information leakage: {str(e)}")
            
        return {'information_leakage': information_leakage}
        
    def _detect_statistical_leakage(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Detect statistical patterns that suggest leakage"""
        statistical_leakage = {
            'constant_features': [],
            'quasi_constant_features': [],
            'single_value_features': []
        }
        
        try:
            for col in df.columns:
                if col == target_column:
                    continue
                    
                series = df[col]
                
                # Constant features (all same value)
                if series.nunique() <= 1:
                    statistical_leakage['constant_features'].append({
                        'feature': col,
                        'unique_values': int(series.nunique()),
                        'severity': 'Medium'
                    })
                    
                # Quasi-constant features (>95% same value)
                elif series.nunique() > 1:
                    most_frequent_pct = series.value_counts().iloc[0] / len(series)
                    if most_frequent_pct >= 0.95:
                        statistical_leakage['quasi_constant_features'].append({
                            'feature': col,
                            'dominant_value_percentage': float(most_frequent_pct * 100),
                            'unique_values': int(series.nunique()),
                            'severity': 'Low'
                        })
                        
                # Features with very few unique values relative to sample size
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.01 and series.nunique() > 1:
                    statistical_leakage['single_value_features'].append({
                        'feature': col,
                        'unique_ratio': float(unique_ratio),
                        'unique_values': int(series.nunique()),
                        'severity': 'Low'
                    })
                    
        except Exception as e:
            self.logger.warning(f"Error detecting statistical leakage: {str(e)}")
            
        return {'statistical_leakage': statistical_leakage}
        
    def _is_perfect_predictor(self, feature_series: pd.Series, target_series: pd.Series) -> bool:
        """Check if a feature is a perfect predictor of the target"""
        try:
            # Align series
            aligned_df = pd.DataFrame({'feature': feature_series, 'target': target_series}).dropna()
            
            if len(aligned_df) < 5:
                return False
                
            # Group by feature value and check target consistency
            grouped = aligned_df.groupby('feature')['target'].nunique()
            
            # If each feature value maps to exactly one target value, it's a perfect predictor
            return (grouped == 1).all()
            
        except Exception:
            return False
            
    def _calculate_information_score(self, feature_series: pd.Series, target_series: pd.Series) -> float:
        """Calculate information score between feature and target"""
        try:
            # Simple correlation-based score for numeric data
            if pd.api.types.is_numeric_dtype(feature_series) and pd.api.types.is_numeric_dtype(target_series):
                corr, _ = pearsonr(feature_series.fillna(0), target_series.fillna(0))
                return abs(corr) if not np.isnan(corr) else 0.0
            else:
                # For categorical data, use a simple uniqueness-based score
                aligned_df = pd.DataFrame({'feature': feature_series, 'target': target_series}).dropna()
                if len(aligned_df) < 5:
                    return 0.0
                    
                # Calculate how well feature values predict target
                grouped = aligned_df.groupby('feature')['target'].nunique()
                perfect_predictions = (grouped == 1).sum()
                total_groups = len(grouped)
                
                return perfect_predictions / total_groups if total_groups > 0 else 0.0
                
        except Exception:
            return 0.0
            
    def _has_suspicious_name(self, feature_name: str, target_name: str) -> bool:
        """Check if feature name suggests target information leakage"""
        feature_lower = feature_name.lower()
        target_lower = target_name.lower()
        
        # Check if feature name contains target name
        if target_lower in feature_lower and feature_name != target_name:
            return True
            
        # Check for suspicious keywords
        suspicious_keywords = [
            'result', 'outcome', 'prediction', 'label', 'class', 'category',
            'decision', 'verdict', 'conclusion', 'answer', 'response',
            'target', 'goal', 'objective', 'success', 'failure'
        ]
        
        return any(keyword in feature_lower for keyword in suspicious_keywords)
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on leakage detection results"""
        recommendations = []
        
        # Perfect correlations
        perfect_corrs = results.get('perfect_correlations', [])
        if perfect_corrs:
            recommendations.append(f"🚨 {len(perfect_corrs)} perfect correlations detected!")
            for corr in perfect_corrs[:3]:
                recommendations.append(f"  • Remove {corr['feature2']} (correlation with {corr['feature1']}: {corr['correlation']:.3f})")
            if len(perfect_corrs) > 3:
                recommendations.append(f"  • ... and {len(perfect_corrs) - 3} more correlations")
                
        # Duplicate features
        duplicates = results.get('duplicate_features', [])
        if duplicates:
            recommendations.append(f"🔄 {len(duplicates)} duplicate/similar features detected!")
            for dup in duplicates[:3]:
                recommendations.append(f"  • {dup['recommendation']} (similarity: {dup['similarity']:.3f})")
                
        # Target leakage
        target_leakage = results.get('target_leakage', {})
        perfect_predictors = target_leakage.get('perfect_predictors', [])
        if perfect_predictors:
            recommendations.append(f"⚠️ {len(perfect_predictors)} perfect predictors detected!")
            for pred in perfect_predictors:
                recommendations.append(f"  • Remove {pred['feature']} - causes target leakage")
                
        high_info = target_leakage.get('high_information_features', [])
        if high_info:
            recommendations.append(f"📊 {len(high_info)} high-information features detected!")
            recommendations.append("  • Review these features for potential leakage")
            
        # Temporal leakage
        temporal = results.get('temporal_leakage', [])
        critical_temporal = [t for t in temporal if t.get('severity') == 'Critical']
        if critical_temporal:
            recommendations.append(f"⏰ {len(critical_temporal)} critical temporal leakage issues!")
            for temp in critical_temporal:
                recommendations.append(f"  • Fix {temp['feature']}: {temp['issue']}")
                
        # Information leakage
        info_leakage = results.get('information_leakage', {})
        high_mi = info_leakage.get('high_mutual_info_features', [])
        if high_mi:
            recommendations.append(f"🔍 {len(high_mi)} features with high mutual information!")
            recommendations.append("  • Investigate these features for potential leakage:")
            for mi in high_mi[:3]:
                recommendations.append(f"    - {mi['feature']} (MI: {mi['normalized_mi']:.3f})")
                
        # Statistical leakage
        stat_leakage = results.get('statistical_leakage', {})
        constants = stat_leakage.get('constant_features', [])
        if constants:
            recommendations.append(f"🔒 {len(constants)} constant features detected!")
            recommendations.append("  • Remove constant features as they provide no information")
            
        # General recommendations
        if not recommendations:
            recommendations.append("✅ No significant leakage patterns detected!")
            recommendations.append("  • Data appears clean from leakage perspective")
        else:
            recommendations.append("\n📋 General Recommendations:")
            recommendations.append("  • Carefully review flagged features before model training")
            recommendations.append("  • Consider feature engineering to reduce leakage")
            recommendations.append("  • Implement proper train/validation/test splits")
            recommendations.append("  • Use time-based splits for temporal data")
            
        return recommendations
        
    def _calculate_leakage_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall leakage risk score (0-100, lower is better)"""
        try:
            score = 0.0
            max_score = 100.0
            
            # Perfect correlations penalty
            perfect_corrs = len(results.get('perfect_correlations', []))
            score += min(perfect_corrs * 15, 30)  # Up to 30 points
            
            # Duplicate features penalty
            duplicates = len(results.get('duplicate_features', []))
            score += min(duplicates * 10, 25)  # Up to 25 points
            
            # Target leakage penalty
            target_leakage = results.get('target_leakage', {})
            perfect_predictors = len(target_leakage.get('perfect_predictors', []))
            score += perfect_predictors * 20  # 20 points each
            
            high_info = len(target_leakage.get('high_information_features', []))
            score += min(high_info * 5, 20)  # Up to 20 points
            
            # Temporal leakage penalty
            temporal = results.get('temporal_leakage', [])
            critical_temporal = len([t for t in temporal if t.get('severity') == 'Critical'])
            score += critical_temporal * 15  # 15 points each
            
            # Information leakage penalty
            info_leakage = results.get('information_leakage', {})
            high_mi = len(info_leakage.get('high_mutual_info_features', []))
            score += min(high_mi * 8, 15)  # Up to 15 points
            
            return min(score, max_score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating leakage score: {str(e)}")
            return 50.0  # Default moderate risk score