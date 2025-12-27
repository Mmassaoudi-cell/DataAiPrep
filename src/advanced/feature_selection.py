"""
Advanced Feature Selection Module

Features:
- Boruta: All-relevant feature selection using Random Forest
- RFE with Cross-Validation: Recursive Feature Elimination
- LASSO/Elastic Net: Regularization-based selection
- mRMR: Minimum Redundancy Maximum Relevance
- Variance Threshold: Remove low-variance features
- Correlation Filter: Remove highly correlated features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE, RFECV, SelectFromModel, VarianceThreshold,
    mutual_info_classif, mutual_info_regression
)
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureSelectionResult:
    """Container for feature selection results"""
    method: str
    selected_features: List[str]
    feature_scores: Dict[str, float]
    n_features_selected: int
    n_features_original: int
    selection_ratio: float


class AdvancedFeatureSelector:
    """
    Advanced feature selection using multiple methods.
    
    Methods:
    - Boruta: Shadow features comparison with Random Forest
    - RFE/RFECV: Recursive Feature Elimination with optional CV
    - LASSO/Elastic Net: L1/L2 regularization-based selection
    - mRMR: Minimum Redundancy Maximum Relevance
    - Variance Threshold: Remove constant/quasi-constant features
    - Correlation Filter: Remove redundant correlated features
    
    Features:
    - Consensus selection across multiple methods
    - Feature importance ranking
    - Optimal feature subset identification
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the feature selector.
        
        Args:
            n_estimators: Number of estimators for tree-based methods
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.results_ = {}
        
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = None,
        n_features: int = None,
        consensus_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run feature selection using multiple methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            methods: List of methods to use (default: all)
            n_features: Target number of features (optional)
            consensus_threshold: Fraction of methods that must agree
            
        Returns:
            Dictionary with selection results
        """
        if methods is None:
            methods = ['boruta', 'rfe', 'lasso', 'mrmr', 'variance', 'correlation']
        
        results = {
            'summary': {},
            'by_method': {},
            'feature_rankings': {},
            'consensus_features': [],
            'recommendations': []
        }
        
        # Determine task type
        is_classification = y.nunique() <= 20 or y.dtype == 'object'
        
        # Prepare data
        X_clean = X.select_dtypes(include=[np.number]).copy()
        X_clean = X_clean.fillna(X_clean.median())
        
        if y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = pd.Series(le.fit_transform(y), index=y.index)
        else:
            y_encoded = y.copy()
        
        # Track feature votes
        feature_votes = {col: 0 for col in X_clean.columns}
        total_methods = 0
        
        # Run each method
        for method in methods:
            try:
                if method == 'boruta':
                    method_result = self._boruta_selection(X_clean, y_encoded, is_classification)
                elif method == 'rfe':
                    method_result = self._rfe_selection(X_clean, y_encoded, is_classification, n_features)
                elif method == 'lasso':
                    method_result = self._lasso_selection(X_clean, y_encoded, is_classification)
                elif method == 'elastic_net':
                    method_result = self._elastic_net_selection(X_clean, y_encoded, is_classification)
                elif method == 'mrmr':
                    method_result = self._mrmr_selection(X_clean, y_encoded, is_classification, n_features)
                elif method == 'variance':
                    method_result = self._variance_selection(X_clean)
                elif method == 'correlation':
                    method_result = self._correlation_selection(X_clean, threshold=0.95)
                else:
                    continue
                
                results['by_method'][method] = method_result
                total_methods += 1
                
                # Count votes
                for feature in method_result['selected_features']:
                    if feature in feature_votes:
                        feature_votes[feature] += 1
                        
            except Exception as e:
                results['by_method'][method] = {'error': str(e)}
        
        # Calculate consensus features
        if total_methods > 0:
            min_votes = int(total_methods * consensus_threshold)
            results['consensus_features'] = [
                feat for feat, votes in feature_votes.items()
                if votes >= max(min_votes, 1)
            ]
        
        # Calculate feature rankings (average rank across methods)
        all_rankings = {}
        for method, method_result in results['by_method'].items():
            if 'feature_scores' in method_result:
                for feat, score in method_result['feature_scores'].items():
                    if feat not in all_rankings:
                        all_rankings[feat] = []
                    all_rankings[feat].append(score)
        
        results['feature_rankings'] = {
            feat: float(np.mean(scores))
            for feat, scores in all_rankings.items()
        }
        results['feature_rankings'] = dict(sorted(
            results['feature_rankings'].items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Summary
        results['summary'] = {
            'original_features': len(X_clean.columns),
            'methods_used': total_methods,
            'consensus_features_count': len(results['consensus_features']),
            'top_features': list(results['feature_rankings'].keys())[:10]
        }
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results, X_clean)
        
        self.results_ = results
        return results
    
    def _boruta_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Boruta feature selection using shadow features.
        
        Boruta creates shadow features (shuffled copies of original features)
        and compares feature importance against the maximum shadow importance.
        """
        n_iterations = 20
        feature_hits = {col: 0 for col in X.columns}
        
        for iteration in range(n_iterations):
            # Create shadow features
            X_shadow = X.apply(lambda x: np.random.permutation(x.values))
            X_shadow.columns = [f'shadow_{col}' for col in X.columns]
            
            # Combine original and shadow features
            X_combined = pd.concat([X, X_shadow], axis=1)
            
            # Train Random Forest
            if is_classification:
                rf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state + iteration,
                    n_jobs=self.n_jobs,
                    max_depth=7
                )
            else:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state + iteration,
                    n_jobs=self.n_jobs,
                    max_depth=7
                )
            
            rf.fit(X_combined, y)
            importances = dict(zip(X_combined.columns, rf.feature_importances_))
            
            # Get max shadow importance
            shadow_importances = [
                imp for col, imp in importances.items()
                if col.startswith('shadow_')
            ]
            max_shadow = max(shadow_importances) if shadow_importances else 0
            
            # Count hits (feature importance > max shadow)
            for col in X.columns:
                if importances.get(col, 0) > max_shadow:
                    feature_hits[col] += 1
        
        # Select features that beat shadow in majority of iterations
        threshold = n_iterations * 0.5
        selected_features = [
            col for col, hits in feature_hits.items()
            if hits >= threshold
        ]
        
        # Normalize scores
        feature_scores = {
            col: hits / n_iterations
            for col, hits in feature_hits.items()
        }
        
        return {
            'method': 'boruta',
            'selected_features': selected_features,
            'feature_scores': dict(sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'n_features_selected': len(selected_features),
            'n_iterations': n_iterations,
            'threshold': threshold
        }
    
    def _rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool,
        n_features: int = None
    ) -> Dict[str, Any]:
        """
        Recursive Feature Elimination with Cross-Validation.
        """
        if is_classification:
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                max_depth=5
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                max_depth=5
            )
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        if n_features is None:
            # Use RFECV to find optimal number of features
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=cv,
                scoring='accuracy' if is_classification else 'neg_mean_squared_error',
                min_features_to_select=1,
                n_jobs=self.n_jobs
            )
        else:
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=1
            )
        
        selector.fit(X, y)
        
        selected_features = X.columns[selector.support_].tolist()
        
        # Get feature rankings (lower is better in RFE)
        feature_scores = {
            col: float(1 / rank) if rank > 0 else 0
            for col, rank in zip(X.columns, selector.ranking_)
        }
        
        result = {
            'method': 'rfe',
            'selected_features': selected_features,
            'feature_scores': dict(sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'n_features_selected': len(selected_features),
            'rankings': dict(zip(X.columns, selector.ranking_.tolist()))
        }
        
        if hasattr(selector, 'cv_results_'):
            result['cv_scores'] = selector.cv_results_['mean_test_score'].tolist()
            result['optimal_n_features'] = selector.n_features_
        
        return result
    
    def _lasso_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        LASSO (L1 regularization) based feature selection.
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if is_classification:
            # Use Logistic Regression with L1 penalty
            model = LogisticRegressionCV(
                cv=5,
                penalty='l1',
                solver='saga',
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=self.n_jobs
            )
            model.fit(X_scaled, y)
            coefs = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            model = LassoCV(
                cv=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                max_iter=1000
            )
            model.fit(X_scaled, y)
            coefs = np.abs(model.coef_)
        
        # Select features with non-zero coefficients
        feature_scores = dict(zip(X.columns, coefs.flatten()))
        selected_features = [
            col for col, score in feature_scores.items()
            if score > 1e-5
        ]
        
        # Normalize scores
        max_score = max(feature_scores.values()) if feature_scores.values() else 1
        feature_scores = {
            col: score / max_score if max_score > 0 else 0
            for col, score in feature_scores.items()
        }
        
        return {
            'method': 'lasso',
            'selected_features': selected_features,
            'feature_scores': dict(sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'n_features_selected': len(selected_features),
            'regularization_strength': float(model.alpha_) if hasattr(model, 'alpha_') else None
        }
    
    def _elastic_net_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Elastic Net (L1 + L2 regularization) based feature selection.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if is_classification:
            # Use Logistic Regression with Elastic Net
            model = LogisticRegressionCV(
                cv=5,
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[0.1, 0.5, 0.9],
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=self.n_jobs
            )
            model.fit(X_scaled, y)
            coefs = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            model = ElasticNetCV(
                cv=5,
                l1_ratio=[0.1, 0.5, 0.7, 0.9],
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                max_iter=1000
            )
            model.fit(X_scaled, y)
            coefs = np.abs(model.coef_)
        
        feature_scores = dict(zip(X.columns, coefs.flatten()))
        selected_features = [
            col for col, score in feature_scores.items()
            if score > 1e-5
        ]
        
        max_score = max(feature_scores.values()) if feature_scores.values() else 1
        feature_scores = {
            col: score / max_score if max_score > 0 else 0
            for col, score in feature_scores.items()
        }
        
        return {
            'method': 'elastic_net',
            'selected_features': selected_features,
            'feature_scores': dict(sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'n_features_selected': len(selected_features),
            'l1_ratio': float(model.l1_ratio_) if hasattr(model, 'l1_ratio_') else None
        }
    
    def _mrmr_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool,
        n_features: int = None
    ) -> Dict[str, Any]:
        """
        Minimum Redundancy Maximum Relevance (mRMR) feature selection.
        
        Selects features that are:
        1. Highly relevant to the target (Maximum Relevance)
        2. Minimally redundant with each other (Minimum Redundancy)
        """
        if n_features is None:
            n_features = min(20, len(X.columns))
        
        # Calculate relevance (mutual information with target)
        if is_classification:
            relevance = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            relevance = mutual_info_regression(X, y, random_state=self.random_state)
        
        relevance_dict = dict(zip(X.columns, relevance))
        
        # Calculate redundancy matrix (correlation between features)
        correlation_matrix = X.corr().abs()
        
        # mRMR selection
        selected_features = []
        remaining_features = list(X.columns)
        feature_scores = {}
        
        for i in range(min(n_features, len(remaining_features))):
            best_feature = None
            best_score = -np.inf
            
            for feature in remaining_features:
                # Relevance
                rel = relevance_dict[feature]
                
                # Redundancy (average correlation with already selected features)
                if selected_features:
                    red = correlation_matrix.loc[feature, selected_features].mean()
                else:
                    red = 0
                
                # mRMR score = Relevance - Redundancy
                score = rel - red
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                feature_scores[best_feature] = float(best_score)
        
        # Normalize scores
        if feature_scores:
            min_score = min(feature_scores.values())
            max_score = max(feature_scores.values())
            score_range = max_score - min_score if max_score != min_score else 1
            feature_scores = {
                col: (score - min_score) / score_range
                for col, score in feature_scores.items()
            }
        
        return {
            'method': 'mrmr',
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'n_features_selected': len(selected_features),
            'relevance_scores': {k: float(v) for k, v in relevance_dict.items()}
        }
    
    def _variance_selection(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Remove features with low variance.
        """
        # Scale first to make variance comparable
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        variances = X_scaled.var()
        
        # Select features above threshold
        selected_features = variances[variances >= threshold].index.tolist()
        
        # Normalize variance scores
        max_var = variances.max() if variances.max() > 0 else 1
        feature_scores = (variances / max_var).to_dict()
        
        return {
            'method': 'variance',
            'selected_features': selected_features,
            'feature_scores': dict(sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'n_features_selected': len(selected_features),
            'threshold': threshold,
            'removed_features': [col for col in X.columns if col not in selected_features]
        }
    
    def _correlation_selection(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Remove highly correlated features (keep one from each correlated pair).
        """
        corr_matrix = X.corr().abs()
        
        # Find correlated pairs
        correlated_pairs = []
        features_to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    correlated_pairs.append({
                        'feature1': col_i,
                        'feature2': col_j,
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
                    # Remove the second feature
                    features_to_remove.add(col_j)
        
        selected_features = [col for col in X.columns if col not in features_to_remove]
        
        # Score based on average correlation (lower is better -> invert)
        avg_correlations = corr_matrix.mean()
        feature_scores = {
            col: float(1 - avg_correlations[col])
            for col in X.columns
        }
        
        return {
            'method': 'correlation',
            'selected_features': selected_features,
            'feature_scores': dict(sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'n_features_selected': len(selected_features),
            'threshold': threshold,
            'correlated_pairs': correlated_pairs,
            'removed_features': list(features_to_remove)
        }
    
    def _generate_recommendations(
        self,
        results: Dict[str, Any],
        X: pd.DataFrame
    ) -> List[Dict[str, str]]:
        """Generate feature selection recommendations"""
        recommendations = []
        
        n_original = len(X.columns)
        n_consensus = len(results.get('consensus_features', []))
        
        # Overall recommendation
        if n_consensus < n_original * 0.3:
            recommendations.append({
                'category': 'Feature Reduction',
                'severity': 'high',
                'message': f'Significant feature reduction possible: {n_consensus}/{n_original} features selected by consensus'
            })
        
        # Method-specific recommendations
        for method, method_result in results.get('by_method', {}).items():
            if 'error' in method_result:
                continue
            
            n_selected = method_result.get('n_features_selected', 0)
            
            if method == 'variance':
                removed = method_result.get('removed_features', [])
                if removed:
                    recommendations.append({
                        'category': 'Low Variance',
                        'severity': 'medium',
                        'message': f'{len(removed)} constant/low-variance features detected: {", ".join(removed[:5])}'
                    })
            
            elif method == 'correlation':
                pairs = method_result.get('correlated_pairs', [])
                if pairs:
                    recommendations.append({
                        'category': 'Redundancy',
                        'severity': 'medium',
                        'message': f'{len(pairs)} highly correlated feature pairs detected'
                    })
        
        # Top features recommendation
        top_features = results.get('summary', {}).get('top_features', [])
        if top_features:
            recommendations.append({
                'category': 'Top Features',
                'severity': 'info',
                'message': f'Most important features: {", ".join(top_features[:5])}'
            })
        
        return recommendations
    
    def get_optimal_features(self, min_methods: int = 2) -> List[str]:
        """
        Get features selected by at least min_methods.
        
        Args:
            min_methods: Minimum number of methods that must select the feature
            
        Returns:
            List of optimal features
        """
        if not self.results_:
            return []
        
        feature_votes = {}
        for method, result in self.results_.get('by_method', {}).items():
            if 'selected_features' in result:
                for feat in result['selected_features']:
                    feature_votes[feat] = feature_votes.get(feat, 0) + 1
        
        return [
            feat for feat, votes in feature_votes.items()
            if votes >= min_methods
        ]
    
    def generate_report(self) -> str:
        """Generate human-readable feature selection report"""
        if not self.results_:
            return "No results available. Run select_features() first."
        
        report = []
        report.append("=" * 60)
        report.append("ADVANCED FEATURE SELECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        summary = self.results_.get('summary', {})
        report.append(f"Original Features: {summary.get('original_features', 'N/A')}")
        report.append(f"Methods Used: {summary.get('methods_used', 'N/A')}")
        report.append(f"Consensus Features: {summary.get('consensus_features_count', 'N/A')}")
        report.append("")
        
        # Method results
        report.append("-" * 60)
        report.append("RESULTS BY METHOD")
        report.append("-" * 60)
        
        for method, result in self.results_.get('by_method', {}).items():
            if 'error' in result:
                report.append(f"\n{method.upper()}: Error - {result['error']}")
            else:
                n_selected = result.get('n_features_selected', 0)
                report.append(f"\n{method.upper()}: {n_selected} features selected")
                
                top_features = list(result.get('feature_scores', {}).keys())[:5]
                if top_features:
                    report.append(f"  Top features: {', '.join(top_features)}")
        
        # Consensus features
        report.append("")
        report.append("-" * 60)
        report.append("CONSENSUS FEATURES")
        report.append("-" * 60)
        
        consensus = self.results_.get('consensus_features', [])
        if consensus:
            for feat in consensus[:20]:
                report.append(f"  • {feat}")
            if len(consensus) > 20:
                report.append(f"  ... and {len(consensus) - 20} more")
        else:
            report.append("  No consensus features found")
        
        # Recommendations
        report.append("")
        report.append("-" * 60)
        report.append("RECOMMENDATIONS")
        report.append("-" * 60)
        
        for rec in self.results_.get('recommendations', []):
            report.append(f"\n[{rec['severity'].upper()}] {rec['category']}")
            report.append(f"  {rec['message']}")
        
        return "\n".join(report)

