"""
SHAP Integration Module

Provides true SHAP (SHapley Additive exPlanations) integration for
model-agnostic feature importance and explanation.

Features:
- SHAP Feature Importance (global and local)
- SHAP Summary Plots data
- SHAP Dependence data
- SHAP Force Plot data
- Feature contribution tracking across data slices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results"""
    feature: str
    shap_value: float
    feature_value: float
    contribution: str  # 'positive', 'negative'


class SHAPExplainer:
    """
    SHAP-based model explainability.
    
    Uses SHAP library when available, falls back to approximation otherwise.
    
    Features:
    - Global feature importance using mean |SHAP values|
    - Local explanations for individual predictions
    - Feature interaction detection
    - Summary statistics for explanations
    - Data slice analysis with SHAP
    """
    
    def __init__(
        self,
        random_state: int = 42,
        max_samples: int = 1000
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            random_state: Random seed for reproducibility
            max_samples: Maximum samples for SHAP calculation
        """
        self.random_state = random_state
        self.max_samples = max_samples
        self._shap_available = self._check_shap()
        self.results_ = None
        self._model = None
        self._explainer = None
        self._shap_values = None
        
    def _check_shap(self) -> bool:
        """Check if SHAP library is available"""
        try:
            import shap
            self._shap = shap
            return True
        except ImportError:
            return False
    
    def explain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model=None,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for the data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model: Pre-trained model (optional, will train if not provided)
            feature_names: Feature names (uses X.columns if not provided)
            
        Returns:
            Dictionary with SHAP explanation results
        """
        results = {
            'summary': {},
            'global_importance': {},
            'feature_effects': {},
            'interaction_effects': {},
            'sample_explanations': [],
            'shap_library_used': self._shap_available
        }
        
        # Prepare data
        X_clean = X.select_dtypes(include=[np.number]).copy()
        X_clean = X_clean.fillna(X_clean.median())
        feature_names = feature_names or X_clean.columns.tolist()
        
        # Determine task type
        is_classification = y.nunique() <= 20 or y.dtype == 'object'
        
        # Train model if not provided
        if model is None:
            model = self._train_model(X_clean, y, is_classification)
        self._model = model
        
        # Sample data if too large
        if len(X_clean) > self.max_samples:
            sample_idx = np.random.choice(
                len(X_clean), 
                self.max_samples, 
                replace=False
            )
            X_sample = X_clean.iloc[sample_idx]
        else:
            X_sample = X_clean
        
        if self._shap_available:
            results = self._compute_shap_values(
                X_sample, model, feature_names, is_classification, results
            )
        else:
            results = self._compute_approximate_shap(
                X_sample, y.iloc[X_sample.index] if len(X_sample) < len(y) else y,
                model, feature_names, is_classification, results
            )
        
        # Generate summary
        results['summary'] = {
            'n_features': len(feature_names),
            'n_samples_analyzed': len(X_sample),
            'model_type': type(model).__name__,
            'shap_method': 'shap_library' if self._shap_available else 'approximation'
        }
        
        self.results_ = results
        return results
    
    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool
    ):
        """Train a model for SHAP analysis"""
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        model.fit(X, y)
        return model
    
    def _compute_shap_values(
        self,
        X: pd.DataFrame,
        model,
        feature_names: List[str],
        is_classification: bool,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute SHAP values using the SHAP library"""
        shap = self._shap
        
        try:
            # Create SHAP explainer based on model type
            if hasattr(model, 'estimators_'):
                # Tree-based model
                explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models
                background = shap.sample(X, min(100, len(X)))
                explainer = shap.KernelExplainer(
                    model.predict_proba if is_classification else model.predict,
                    background
                )
            
            self._explainer = explainer
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # For classification, take the mean absolute across classes
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            
            self._shap_values = shap_values
            
            # Global importance (mean |SHAP value| per feature)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            global_importance = dict(zip(feature_names, mean_abs_shap))
            results['global_importance'] = dict(sorted(
                global_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Feature effects (direction and magnitude)
            for i, feat in enumerate(feature_names):
                feat_shap = shap_values[:, i]
                feat_values = X.iloc[:, i].values
                
                # Correlation between feature value and SHAP value
                if np.std(feat_values) > 0 and np.std(feat_shap) > 0:
                    corr = np.corrcoef(feat_values, feat_shap)[0, 1]
                else:
                    corr = 0
                
                results['feature_effects'][feat] = {
                    'mean_shap': float(np.mean(feat_shap)),
                    'mean_abs_shap': float(np.mean(np.abs(feat_shap))),
                    'std_shap': float(np.std(feat_shap)),
                    'direction': 'positive' if corr > 0.1 else ('negative' if corr < -0.1 else 'mixed'),
                    'correlation': float(corr) if not np.isnan(corr) else 0
                }
            
            # Sample explanations (top influential samples)
            sample_importance = np.abs(shap_values).sum(axis=1)
            top_sample_idx = np.argsort(sample_importance)[-10:][::-1]
            
            for idx in top_sample_idx:
                sample_explanation = {
                    'sample_index': int(X.index[idx]),
                    'total_shap_magnitude': float(sample_importance[idx]),
                    'top_contributors': []
                }
                
                sample_shap = shap_values[idx]
                sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:5]
                
                for feat_idx in sorted_idx:
                    sample_explanation['top_contributors'].append({
                        'feature': feature_names[feat_idx],
                        'shap_value': float(sample_shap[feat_idx]),
                        'feature_value': float(X.iloc[idx, feat_idx])
                    })
                
                results['sample_explanations'].append(sample_explanation)
            
            # Interaction effects (for tree-based models)
            if hasattr(explainer, 'shap_interaction_values'):
                try:
                    # Only compute for a subset due to computational cost
                    X_subset = X.iloc[:min(100, len(X))]
                    interaction_values = explainer.shap_interaction_values(X_subset)
                    
                    if isinstance(interaction_values, list):
                        interaction_values = np.array(interaction_values).mean(axis=0)
                    
                    # Find strongest interactions
                    mean_interactions = np.abs(interaction_values).mean(axis=0)
                    
                    interactions = []
                    for i in range(len(feature_names)):
                        for j in range(i + 1, len(feature_names)):
                            if mean_interactions[i, j] > 0.01:
                                interactions.append({
                                    'feature1': feature_names[i],
                                    'feature2': feature_names[j],
                                    'interaction_strength': float(mean_interactions[i, j])
                                })
                    
                    results['interaction_effects'] = sorted(
                        interactions,
                        key=lambda x: x['interaction_strength'],
                        reverse=True
                    )[:20]
                except Exception:
                    pass
                    
        except Exception as e:
            results['error'] = f"SHAP calculation failed: {str(e)}"
            # Fall back to approximation
            results = self._compute_approximate_shap(
                X, None, model, feature_names, is_classification, results
            )
        
        return results
    
    def _compute_approximate_shap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        feature_names: List[str],
        is_classification: bool,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute approximate SHAP values using permutation importance.
        
        This is a fallback when the SHAP library is not available.
        """
        from sklearn.inspection import permutation_importance
        
        # Use permutation importance as approximation
        if y is not None:
            perm_importance = permutation_importance(
                model, X, y,
                n_repeats=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Normalize to approximate SHAP-like values
            importance_mean = perm_importance.importances_mean
            max_imp = importance_mean.max() if importance_mean.max() > 0 else 1
            normalized_importance = importance_mean / max_imp
            
            results['global_importance'] = dict(sorted(
                zip(feature_names, normalized_importance),
                key=lambda x: x[1],
                reverse=True
            ))
            
            results['global_importance'] = {
                k: float(v) for k, v in results['global_importance'].items()
            }
        
        # Tree-based feature importance as additional signal
        if hasattr(model, 'feature_importances_'):
            tree_importance = dict(zip(feature_names, model.feature_importances_))
            results['tree_importance'] = dict(sorted(
                tree_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            results['tree_importance'] = {
                k: float(v) for k, v in results['tree_importance'].items()
            }
            
            # If permutation failed, use tree importance as global
            if not results.get('global_importance'):
                results['global_importance'] = results['tree_importance']
        
        # Feature effects based on correlation with predictions
        predictions = model.predict(X)
        
        for feat in feature_names:
            feat_values = X[feat].values
            
            if np.std(feat_values) > 0 and np.std(predictions) > 0:
                corr = np.corrcoef(feat_values, predictions)[0, 1]
            else:
                corr = 0
            
            results['feature_effects'][feat] = {
                'correlation_with_prediction': float(corr) if not np.isnan(corr) else 0,
                'direction': 'positive' if corr > 0.1 else ('negative' if corr < -0.1 else 'mixed'),
                'importance': float(results['global_importance'].get(feat, 0))
            }
        
        results['note'] = "SHAP library not available. Using permutation importance approximation."
        
        return results
    
    def explain_instance(
        self,
        X_instance: pd.DataFrame,
        top_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single instance prediction.
        
        Args:
            X_instance: Single row DataFrame to explain
            top_features: Number of top contributing features
            
        Returns:
            Dictionary with instance explanation
        """
        if self._model is None:
            return {'error': 'No model available. Run explain() first.'}
        
        explanation = {
            'prediction': None,
            'base_value': None,
            'contributions': []
        }
        
        # Get prediction
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(X_instance)[0]
            explanation['prediction'] = {
                'class': int(self._model.predict(X_instance)[0]),
                'probabilities': proba.tolist()
            }
        else:
            explanation['prediction'] = float(self._model.predict(X_instance)[0])
        
        if self._shap_available and self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(X_instance)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Take positive class
                
                shap_values = shap_values.flatten()
                
                if hasattr(self._explainer, 'expected_value'):
                    expected = self._explainer.expected_value
                    if isinstance(expected, (list, np.ndarray)):
                        explanation['base_value'] = float(expected[1])
                    else:
                        explanation['base_value'] = float(expected)
                
                # Get top contributors
                feature_names = X_instance.columns.tolist()
                contributions = list(zip(feature_names, shap_values, X_instance.values.flatten()))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for feat, shap_val, feat_val in contributions[:top_features]:
                    explanation['contributions'].append({
                        'feature': feat,
                        'shap_value': float(shap_val),
                        'feature_value': float(feat_val),
                        'direction': 'increases' if shap_val > 0 else 'decreases'
                    })
                    
            except Exception as e:
                explanation['error'] = str(e)
        else:
            explanation['note'] = "Instance explanation requires SHAP library"
        
        return explanation
    
    def get_dependence_data(
        self,
        feature: str,
        interaction_feature: str = None
    ) -> Dict[str, Any]:
        """
        Get data for SHAP dependence plot.
        
        Args:
            feature: Main feature to plot
            interaction_feature: Optional interaction feature for coloring
            
        Returns:
            Dictionary with dependence plot data
        """
        if self._shap_values is None or self.results_ is None:
            return {'error': 'No SHAP values available. Run explain() first.'}
        
        feature_names = list(self.results_.get('global_importance', {}).keys())
        
        if feature not in feature_names:
            return {'error': f'Feature {feature} not found'}
        
        feat_idx = feature_names.index(feature)
        
        result = {
            'feature': feature,
            'feature_values': [],
            'shap_values': []
        }
        
        # Note: This would need access to the original X data
        # For now, return what we have from the SHAP calculation
        result['shap_statistics'] = self.results_.get('feature_effects', {}).get(feature, {})
        
        return result
    
    def get_summary_data(self) -> Dict[str, Any]:
        """
        Get data for SHAP summary plot.
        
        Returns:
            Dictionary with summary plot data
        """
        if self.results_ is None:
            return {'error': 'No results available. Run explain() first.'}
        
        return {
            'global_importance': self.results_.get('global_importance', {}),
            'feature_effects': self.results_.get('feature_effects', {}),
            'top_features': list(self.results_.get('global_importance', {}).keys())[:20]
        }
    
    def generate_report(self) -> str:
        """Generate human-readable SHAP explanation report"""
        if self.results_ is None:
            return "No results available. Run explain() first."
        
        report = []
        report.append("=" * 60)
        report.append("SHAP EXPLANATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        summary = self.results_.get('summary', {})
        report.append(f"Model: {summary.get('model_type', 'N/A')}")
        report.append(f"Samples Analyzed: {summary.get('n_samples_analyzed', 'N/A')}")
        report.append(f"Method: {summary.get('shap_method', 'N/A')}")
        report.append("")
        
        if 'note' in self.results_:
            report.append(f"⚠️  {self.results_['note']}")
            report.append("")
        
        # Global importance
        report.append("-" * 60)
        report.append("GLOBAL FEATURE IMPORTANCE (Mean |SHAP|)")
        report.append("-" * 60)
        
        for i, (feat, importance) in enumerate(
            list(self.results_.get('global_importance', {}).items())[:15]
        ):
            bar_length = int(importance * 40)
            bar = "█" * bar_length
            report.append(f"  {i+1:2d}. {feat:25s} {importance:.4f} {bar}")
        
        report.append("")
        
        # Feature effects
        report.append("-" * 60)
        report.append("FEATURE EFFECTS")
        report.append("-" * 60)
        
        for feat, effect in list(self.results_.get('feature_effects', {}).items())[:10]:
            direction = effect.get('direction', 'unknown')
            direction_symbol = "↑" if direction == 'positive' else ("↓" if direction == 'negative' else "↔")
            report.append(f"\n  {feat}:")
            report.append(f"    Direction: {direction_symbol} {direction}")
            if 'correlation' in effect:
                report.append(f"    Correlation: {effect['correlation']:.3f}")
        
        # Interactions
        interactions = self.results_.get('interaction_effects', [])
        if interactions:
            report.append("")
            report.append("-" * 60)
            report.append("TOP FEATURE INTERACTIONS")
            report.append("-" * 60)
            
            for interaction in interactions[:5]:
                report.append(f"  {interaction['feature1']} × {interaction['feature2']}: {interaction['interaction_strength']:.4f}")
        
        return "\n".join(report)

