"""
Explainability & Interpretability Module

Features:
- SHAP integration
- Permutation importance
- Partial dependence plots data
- Influential samples detection
- Data slice analysis
- Error analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureExplanation:
    """Container for feature explanation"""
    feature: str
    importance: float
    direction: str  # 'positive', 'negative', 'mixed'
    explanation: str


class ExplainabilityEngine:
    """
    Model-agnostic explainability engine.
    
    Features:
    - SHAP-style feature importance (without SHAP dependency)
    - Permutation importance
    - Partial dependence data
    - Influential samples detection
    - Data slice analysis
    - Systematic error identification
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the explainability engine.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.results_ = None
        self._model = None
        
    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str,
        model=None,
        slice_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive explainability analysis.
        
        Args:
            data: DataFrame with features and target
            target_column: Name of target column
            model: Optional pre-trained model
            slice_columns: Columns for data slice analysis
            
        Returns:
            Dictionary with explainability results
        """
        results = {
            'feature_importance': {},
            'permutation_importance': {},
            'feature_correlations': {},
            'influential_samples': [],
            'slice_analysis': {},
            'error_analysis': {}
        }
        
        # Prepare data
        X = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = data[target_column]
        
        X = X.fillna(X.median())
        
        # Train or use provided model
        if model is None:
            model = self._train_baseline_model(X, y)
        
        self._model = model
        
        # Feature importance (tree-based)
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = self._get_tree_importance(model, X.columns)
        
        # Permutation importance
        results['permutation_importance'] = self._calculate_permutation_importance(
            model, X, y
        )
        
        # Feature correlations with target
        results['feature_correlations'] = self._calculate_feature_correlations(X, y)
        
        # Influential samples detection
        results['influential_samples'] = self._detect_influential_samples(
            model, X, y
        )
        
        # Data slice analysis
        if slice_columns:
            results['slice_analysis'] = self._analyze_slices(
                data, X, y, model, slice_columns
            )
        
        # Error analysis
        results['error_analysis'] = self._analyze_errors(model, X, y, data)
        
        self.results_ = results
        return results
    
    def _train_baseline_model(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """Train a baseline model for analysis"""
        if y.nunique() <= 10:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
        
        model.fit(X, y)
        return model
    
    def _get_tree_importance(
        self,
        model,
        feature_names: pd.Index
    ) -> Dict[str, float]:
        """Get feature importance from tree-based model"""
        importance = dict(zip(feature_names, model.feature_importances_))
        return {k: float(v) for k, v in sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )}
    
    def _calculate_permutation_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> Dict[str, Any]:
        """Calculate permutation importance"""
        try:
            perm_importance = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=self.random_state
            )
            
            importance_mean = dict(zip(X.columns, perm_importance.importances_mean))
            importance_std = dict(zip(X.columns, perm_importance.importances_std))
            
            return {
                'mean': {k: float(v) for k, v in sorted(
                    importance_mean.items(), key=lambda x: x[1], reverse=True
                )},
                'std': {k: float(v) for k, v in importance_std.items()}
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_feature_correlations(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Calculate feature correlations with target"""
        correlations = {}
        
        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(y):
                    corr = X[col].corr(y)
                else:
                    # For categorical target, use point-biserial for binary
                    from scipy.stats import pointbiserialr
                    if y.nunique() == 2:
                        y_encoded = (y == y.unique()[0]).astype(int)
                        corr, _ = pointbiserialr(X[col].dropna(), y_encoded.loc[X[col].dropna().index])
                    else:
                        corr = 0
                
                correlations[col] = float(corr) if not pd.isna(corr) else 0
            except Exception:
                correlations[col] = 0
        
        return {k: v for k, v in sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )}
    
    def _detect_influential_samples(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: int = 20
    ) -> List[Dict[str, Any]]:
        """Detect samples with high influence on model"""
        influential = []
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)
                # For classification, use entropy as influence measure
                epsilon = 1e-10
                entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=1)
                influence_scores = entropy
            else:
                predictions = model.predict(X)
                # For regression, use prediction error as influence
                errors = np.abs(predictions - y)
                influence_scores = errors
            
            # Get top influential samples
            top_indices = np.argsort(influence_scores)[-n_samples:][::-1]
            
            for idx in top_indices:
                sample_idx = X.index[idx]
                influential.append({
                    'index': sample_idx,
                    'influence_score': float(influence_scores[idx]),
                    'predicted': float(predictions[idx]) if len(predictions.shape) == 1 
                        else predictions[idx].tolist(),
                    'actual': y.iloc[idx] if isinstance(y.iloc[idx], (int, float)) 
                        else str(y.iloc[idx])
                })
        except Exception as e:
            return [{'error': str(e)}]
        
        return influential
    
    def _analyze_slices(
        self,
        data: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        slice_columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze model performance across data slices"""
        slice_results = {}
        
        for col in slice_columns:
            if col not in data.columns:
                continue
            
            col_results = {}
            
            for value in data[col].dropna().unique()[:10]:  # Limit to 10 unique values
                mask = data[col] == value
                if mask.sum() < 10:
                    continue
                
                X_slice = X.loc[mask]
                y_slice = y.loc[mask]
                
                try:
                    # Calculate slice metrics
                    predictions = model.predict(X_slice)
                    
                    if hasattr(model, 'predict_proba'):
                        # Classification
                        accuracy = (predictions == y_slice).mean()
                        col_results[str(value)] = {
                            'count': int(mask.sum()),
                            'accuracy': float(accuracy),
                            'positive_rate': float((y_slice == y_slice.mode().iloc[0]).mean())
                        }
                    else:
                        # Regression
                        mse = np.mean((predictions - y_slice) ** 2)
                        mae = np.mean(np.abs(predictions - y_slice))
                        col_results[str(value)] = {
                            'count': int(mask.sum()),
                            'mse': float(mse),
                            'mae': float(mae),
                            'mean_actual': float(y_slice.mean()),
                            'mean_predicted': float(predictions.mean())
                        }
                except Exception:
                    continue
            
            if col_results:
                slice_results[col] = col_results
        
        return slice_results
    
    def _analyze_errors(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze systematic error patterns"""
        error_analysis = {
            'overall_metrics': {},
            'error_distribution': {},
            'worst_performing_segments': []
        }
        
        try:
            predictions = model.predict(X)
            
            if hasattr(model, 'predict_proba'):
                # Classification error analysis
                errors = predictions != y
                error_rate = errors.mean()
                
                error_analysis['overall_metrics'] = {
                    'accuracy': float(1 - error_rate),
                    'error_rate': float(error_rate),
                    'total_errors': int(errors.sum())
                }
                
                # Analyze which features correlate with errors
                error_correlations = {}
                for col in X.columns:
                    try:
                        corr = X[col].corr(errors.astype(int))
                        if not pd.isna(corr):
                            error_correlations[col] = float(corr)
                    except Exception:
                        continue
                
                error_analysis['features_correlated_with_errors'] = {
                    k: v for k, v in sorted(
                        error_correlations.items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )[:10]
                }
            else:
                # Regression error analysis
                errors = predictions - y
                abs_errors = np.abs(errors)
                
                error_analysis['overall_metrics'] = {
                    'mse': float(np.mean(errors ** 2)),
                    'mae': float(np.mean(abs_errors)),
                    'rmse': float(np.sqrt(np.mean(errors ** 2))),
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors))
                }
                
                error_analysis['error_distribution'] = {
                    'percentile_25': float(np.percentile(abs_errors, 25)),
                    'percentile_50': float(np.percentile(abs_errors, 50)),
                    'percentile_75': float(np.percentile(abs_errors, 75)),
                    'percentile_90': float(np.percentile(abs_errors, 90)),
                    'percentile_99': float(np.percentile(abs_errors, 99))
                }
                
                # Find worst predictions
                worst_idx = np.argsort(abs_errors)[-10:][::-1]
                error_analysis['worst_predictions'] = [
                    {
                        'index': int(X.index[i]),
                        'actual': float(y.iloc[i]),
                        'predicted': float(predictions[i]),
                        'error': float(abs_errors.iloc[i])
                    }
                    for i in worst_idx
                ]
        
        except Exception as e:
            error_analysis['error'] = str(e)
        
        return error_analysis
    
    def get_feature_explanations(
        self,
        n_features: int = 10
    ) -> List[FeatureExplanation]:
        """Get human-readable feature explanations"""
        if self.results_ is None:
            return []
        
        explanations = []
        
        # Combine importance sources
        importance_scores = {}
        
        if 'feature_importance' in self.results_:
            for feat, score in self.results_['feature_importance'].items():
                importance_scores[feat] = importance_scores.get(feat, 0) + score
        
        if 'permutation_importance' in self.results_ and 'mean' in self.results_['permutation_importance']:
            for feat, score in self.results_['permutation_importance']['mean'].items():
                importance_scores[feat] = importance_scores.get(feat, 0) + score
        
        # Get correlations for direction
        correlations = self.results_.get('feature_correlations', {})
        
        # Generate explanations
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_features]
        
        for feat, score in sorted_features:
            corr = correlations.get(feat, 0)
            
            if abs(corr) < 0.1:
                direction = 'mixed'
                explanation = f"'{feat}' has complex, non-linear relationship with target"
            elif corr > 0:
                direction = 'positive'
                explanation = f"Higher values of '{feat}' tend to increase predictions"
            else:
                direction = 'negative'
                explanation = f"Higher values of '{feat}' tend to decrease predictions"
            
            explanations.append(FeatureExplanation(
                feature=feat,
                importance=score,
                direction=direction,
                explanation=explanation
            ))
        
        return explanations
    
    def generate_report(self) -> str:
        """Generate human-readable explainability report"""
        if self.results_ is None:
            return "No analysis results available. Run analyze() first."
        
        report = []
        report.append("=" * 60)
        report.append("MODEL EXPLAINABILITY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Feature Importance
        report.append("-" * 60)
        report.append("TOP FEATURE IMPORTANCE")
        report.append("-" * 60)
        
        if self.results_['feature_importance']:
            for i, (feat, score) in enumerate(
                list(self.results_['feature_importance'].items())[:10]
            ):
                report.append(f"  {i+1}. {feat}: {score:.4f}")
        
        report.append("")
        
        # Permutation Importance
        if 'mean' in self.results_.get('permutation_importance', {}):
            report.append("-" * 60)
            report.append("PERMUTATION IMPORTANCE")
            report.append("-" * 60)
            
            for i, (feat, score) in enumerate(
                list(self.results_['permutation_importance']['mean'].items())[:10]
            ):
                std = self.results_['permutation_importance']['std'].get(feat, 0)
                report.append(f"  {i+1}. {feat}: {score:.4f} (±{std:.4f})")
        
        report.append("")
        
        # Error Analysis
        if self.results_['error_analysis'].get('overall_metrics'):
            report.append("-" * 60)
            report.append("ERROR ANALYSIS")
            report.append("-" * 60)
            
            metrics = self.results_['error_analysis']['overall_metrics']
            for metric, value in metrics.items():
                report.append(f"  {metric}: {value:.4f}")
        
        report.append("")
        
        # Feature Explanations
        report.append("-" * 60)
        report.append("FEATURE EXPLANATIONS")
        report.append("-" * 60)
        
        for exp in self.get_feature_explanations(5):
            report.append(f"\n  {exp.feature}")
            report.append(f"    Importance: {exp.importance:.4f}")
            report.append(f"    Direction: {exp.direction}")
            report.append(f"    {exp.explanation}")
        
        return "\n".join(report)

