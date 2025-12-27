"""
Ensemble Outlier Detection Module

Features:
- Multi-method ensemble (IQR, Z-score, Isolation Forest, LOF, DBSCAN)
- Contextual outlier detection within groups
- Outlier impact analysis on model performance
- Anomaly explanation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OutlierResult:
    """Container for outlier detection results"""
    index: int
    outlier_score: float
    methods_flagged: List[str]
    explanation: str
    severity: str  # 'low', 'medium', 'high', 'extreme'


class EnsembleOutlierDetector:
    """
    Multi-method ensemble outlier detection.
    
    Methods:
    - IQR (Interquartile Range)
    - Z-score
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - DBSCAN-based
    
    Features:
    - Consensus scoring across methods
    - Contextual outlier detection
    - Impact analysis
    - Anomaly explanation
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        consensus_threshold: float = 0.5,
        contamination: float = 0.05
    ):
        """
        Initialize the ensemble detector.
        
        Args:
            methods: List of methods to use (default: all)
            consensus_threshold: Fraction of methods that must agree
            contamination: Expected proportion of outliers
        """
        self.methods = methods or ['iqr', 'zscore', 'isolation_forest', 'lof', 'dbscan']
        self.consensus_threshold = consensus_threshold
        self.contamination = contamination
        self.results_ = None
        
    def detect(
        self,
        data: pd.DataFrame,
        columns: List[str] = None,
        context_column: str = None
    ) -> Dict[str, Any]:
        """
        Detect outliers using ensemble methods.
        
        Args:
            data: DataFrame to analyze
            columns: Columns to check (default: all numeric)
            context_column: Column for contextual outlier detection
            
        Returns:
            Dictionary with detection results
        """
        # Select numeric columns
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            'summary': {},
            'by_column': {},
            'by_method': {},
            'consensus_outliers': [],
            'outlier_details': []
        }
        
        # Run each method
        method_results = {}
        for method in self.methods:
            try:
                method_results[method] = self._run_method(data, columns, method)
            except Exception as e:
                method_results[method] = {'error': str(e)}
        
        results['by_method'] = method_results
        
        # Compute consensus
        consensus_mask = self._compute_consensus(data, method_results)
        results['consensus_outliers'] = data.index[consensus_mask].tolist()
        
        # Generate column-wise analysis
        for col in columns:
            results['by_column'][col] = self._analyze_column(data, col, method_results)
        
        # Generate outlier details with explanations
        results['outlier_details'] = self._generate_explanations(
            data, columns, method_results, consensus_mask
        )
        
        # Summary statistics
        results['summary'] = {
            'total_rows': len(data),
            'columns_analyzed': len(columns),
            'methods_used': len([m for m in method_results if 'error' not in method_results[m]]),
            'consensus_outliers_count': int(consensus_mask.sum()),
            'consensus_outliers_percentage': float(consensus_mask.sum() / len(data) * 100)
        }
        
        # Contextual analysis if requested
        if context_column and context_column in data.columns:
            results['contextual'] = self._contextual_detection(
                data, columns, context_column
            )
        
        self.results_ = results
        return results
    
    def _run_method(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str
    ) -> Dict[str, Any]:
        """Run a single outlier detection method"""
        numeric_data = data[columns].dropna()
        
        if len(numeric_data) < 10:
            return {'error': 'Insufficient data for outlier detection'}
        
        if method == 'iqr':
            return self._iqr_detection(data, columns)
        elif method == 'zscore':
            return self._zscore_detection(data, columns)
        elif method == 'isolation_forest':
            return self._isolation_forest_detection(data, columns)
        elif method == 'lof':
            return self._lof_detection(data, columns)
        elif method == 'dbscan':
            return self._dbscan_detection(data, columns)
        else:
            return {'error': f'Unknown method: {method}'}
    
    def _iqr_detection(
        self,
        data: pd.DataFrame,
        columns: List[str],
        multiplier: float = 1.5
    ) -> Dict[str, Any]:
        """IQR-based outlier detection"""
        outlier_mask = pd.Series(False, index=data.index)
        column_outliers = {}
        
        for col in columns:
            series = data[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_mask |= col_outliers.fillna(False)
            
            column_outliers[col] = {
                'count': int(col_outliers.sum()),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_indices': data.index[col_outliers.fillna(False)].tolist()[:100]
            }
        
        return {
            'outlier_indices': data.index[outlier_mask].tolist(),
            'outlier_count': int(outlier_mask.sum()),
            'by_column': column_outliers
        }
    
    def _zscore_detection(
        self,
        data: pd.DataFrame,
        columns: List[str],
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Z-score based outlier detection"""
        outlier_mask = pd.Series(False, index=data.index)
        column_outliers = {}
        scores = pd.DataFrame(index=data.index)
        
        for col in columns:
            series = data[col]
            mean = series.mean()
            std = series.std()
            
            if std > 0:
                z_scores = np.abs((series - mean) / std)
                scores[col] = z_scores
                col_outliers = z_scores > threshold
                outlier_mask |= col_outliers.fillna(False)
                
                column_outliers[col] = {
                    'count': int(col_outliers.sum()),
                    'max_zscore': float(z_scores.max()),
                    'outlier_indices': data.index[col_outliers.fillna(False)].tolist()[:100]
                }
        
        return {
            'outlier_indices': data.index[outlier_mask].tolist(),
            'outlier_count': int(outlier_mask.sum()),
            'by_column': column_outliers,
            'z_scores': scores
        }
    
    def _isolation_forest_detection(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """Isolation Forest outlier detection"""
        numeric_data = data[columns].fillna(data[columns].median())
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Fit Isolation Forest
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = model.fit_predict(scaled_data)
        scores = model.score_samples(scaled_data)
        
        outlier_mask = predictions == -1
        
        return {
            'outlier_indices': data.index[outlier_mask].tolist(),
            'outlier_count': int(outlier_mask.sum()),
            'anomaly_scores': pd.Series(scores, index=data.index).to_dict()
        }
    
    def _lof_detection(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """Local Outlier Factor detection"""
        numeric_data = data[columns].fillna(data[columns].median())
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Fit LOF
        n_neighbors = min(20, len(data) // 5)
        model = LocalOutlierFactor(
            n_neighbors=max(5, n_neighbors),
            contamination=self.contamination
        )
        predictions = model.fit_predict(scaled_data)
        scores = model.negative_outlier_factor_
        
        outlier_mask = predictions == -1
        
        return {
            'outlier_indices': data.index[outlier_mask].tolist(),
            'outlier_count': int(outlier_mask.sum()),
            'lof_scores': pd.Series(scores, index=data.index).to_dict()
        }
    
    def _dbscan_detection(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """DBSCAN-based outlier detection (noise points)"""
        numeric_data = data[columns].fillna(data[columns].median())
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Fit DBSCAN
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(scaled_data)
        
        # Noise points are labeled -1
        outlier_mask = labels == -1
        
        return {
            'outlier_indices': data.index[outlier_mask].tolist(),
            'outlier_count': int(outlier_mask.sum()),
            'cluster_labels': pd.Series(labels, index=data.index).to_dict(),
            'n_clusters': int(len(set(labels)) - (1 if -1 in labels else 0))
        }
    
    def _compute_consensus(
        self,
        data: pd.DataFrame,
        method_results: Dict[str, Dict]
    ) -> pd.Series:
        """Compute consensus outliers across methods"""
        vote_counts = pd.Series(0, index=data.index)
        valid_methods = 0
        
        for method, results in method_results.items():
            if 'error' not in results and 'outlier_indices' in results:
                valid_methods += 1
                outlier_indices = results['outlier_indices']
                vote_counts.loc[vote_counts.index.isin(outlier_indices)] += 1
        
        if valid_methods == 0:
            return pd.Series(False, index=data.index)
        
        # Consensus: flagged by at least threshold fraction of methods
        min_votes = max(1, int(valid_methods * self.consensus_threshold))
        return vote_counts >= min_votes
    
    def _analyze_column(
        self,
        data: pd.DataFrame,
        column: str,
        method_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Analyze outliers for a specific column"""
        series = data[column].dropna()
        
        analysis = {
            'statistics': {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'median': float(series.median()),
                'min': float(series.min()),
                'max': float(series.max()),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis())
            },
            'method_counts': {}
        }
        
        for method, results in method_results.items():
            if 'by_column' in results and column in results['by_column']:
                analysis['method_counts'][method] = results['by_column'][column].get('count', 0)
        
        return analysis
    
    def _generate_explanations(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method_results: Dict[str, Dict],
        consensus_mask: pd.Series
    ) -> List[Dict[str, Any]]:
        """Generate explanations for outliers"""
        explanations = []
        
        consensus_indices = data.index[consensus_mask].tolist()[:50]  # Limit to 50
        
        for idx in consensus_indices:
            methods_flagged = []
            for method, results in method_results.items():
                if 'outlier_indices' in results and idx in results['outlier_indices']:
                    methods_flagged.append(method)
            
            # Determine severity
            severity_score = len(methods_flagged) / len(self.methods)
            if severity_score >= 0.8:
                severity = 'extreme'
            elif severity_score >= 0.6:
                severity = 'high'
            elif severity_score >= 0.4:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Generate explanation
            row = data.loc[idx, columns]
            extreme_features = []
            
            for col in columns:
                if pd.notna(row[col]):
                    series = data[col].dropna()
                    z_score = abs((row[col] - series.mean()) / series.std()) if series.std() > 0 else 0
                    percentile = (series < row[col]).sum() / len(series) * 100
                    
                    if z_score > 2:
                        extreme_features.append({
                            'feature': col,
                            'value': float(row[col]),
                            'z_score': float(z_score),
                            'percentile': float(percentile)
                        })
            
            explanation = f"Flagged by {len(methods_flagged)}/{len(self.methods)} methods. "
            if extreme_features:
                top_extreme = sorted(extreme_features, key=lambda x: x['z_score'], reverse=True)[:3]
                explanation += f"Extreme values in: {', '.join(f['feature'] for f in top_extreme)}"
            
            explanations.append({
                'index': idx,
                'severity': severity,
                'methods_flagged': methods_flagged,
                'explanation': explanation,
                'extreme_features': extreme_features[:5]
            })
        
        return explanations
    
    def _contextual_detection(
        self,
        data: pd.DataFrame,
        columns: List[str],
        context_column: str
    ) -> Dict[str, Any]:
        """Detect outliers within specific groups/segments"""
        contextual_results = {}
        
        for group_name, group_data in data.groupby(context_column):
            if len(group_data) < 10:
                continue
            
            group_outliers = []
            
            for col in columns:
                series = group_data[col].dropna()
                if len(series) < 5:
                    continue
                
                # IQR within group
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                
                outlier_mask = (group_data[col] < lower) | (group_data[col] > upper)
                group_outliers.extend(group_data.index[outlier_mask.fillna(False)].tolist())
            
            contextual_results[str(group_name)] = {
                'group_size': len(group_data),
                'outlier_count': len(set(group_outliers)),
                'outlier_indices': list(set(group_outliers))[:50]
            }
        
        return contextual_results
    
    def analyze_impact(
        self,
        data: pd.DataFrame,
        target_column: str,
        model=None
    ) -> Dict[str, Any]:
        """
        Analyze the impact of outlier removal on model performance.
        
        Args:
            data: DataFrame with features and target
            target_column: Name of target column
            model: Optional model to use (default: RandomForest)
            
        Returns:
            Dictionary with impact analysis
        """
        if self.results_ is None:
            self.detect(data)
        
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        consensus_outliers = self.results_.get('consensus_outliers', [])
        
        # Prepare data
        X = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = data[target_column]
        
        X = X.fillna(X.median())
        
        # Determine task type
        if y.nunique() <= 10:
            if model is None:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        else:
            if model is None:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'neg_mean_squared_error'
        
        # Score with all data
        try:
            scores_with = cross_val_score(model, X, y, cv=5, scoring=scoring)
            mean_with = float(np.mean(scores_with))
        except Exception as e:
            return {'error': f'Could not evaluate with outliers: {e}'}
        
        # Score without outliers
        mask = ~data.index.isin(consensus_outliers)
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        
        try:
            scores_without = cross_val_score(model, X_clean, y_clean, cv=5, scoring=scoring)
            mean_without = float(np.mean(scores_without))
        except Exception as e:
            return {'error': f'Could not evaluate without outliers: {e}'}
        
        return {
            'with_outliers': {
                'n_samples': len(X),
                'mean_score': mean_with,
                'std_score': float(np.std(scores_with))
            },
            'without_outliers': {
                'n_samples': len(X_clean),
                'mean_score': mean_without,
                'std_score': float(np.std(scores_without))
            },
            'improvement': mean_without - mean_with,
            'improvement_percentage': (mean_without - mean_with) / abs(mean_with) * 100 if mean_with != 0 else 0,
            'outliers_removed': len(consensus_outliers),
            'scoring_metric': scoring
        }

