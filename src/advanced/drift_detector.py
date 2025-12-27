"""
Data Drift Detection Module

Features:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov Test
- Jensen-Shannon Divergence
- Chi-Square Drift (categorical)
- Concept Drift Detection
- Drift Severity Scoring
- Root Cause Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DriftResult:
    """Container for drift detection result"""
    feature: str
    drift_detected: bool
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    psi_score: Optional[float]
    ks_statistic: Optional[float]
    ks_pvalue: Optional[float]
    js_divergence: Optional[float]


class DriftAnalyzer:
    """
    Comprehensive data drift detection and monitoring.
    
    Methods:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov Test
    - Jensen-Shannon Divergence
    - Chi-Square Test for categorical features
    - Concept drift detection
    
    Features:
    - Multi-method drift detection
    - Severity scoring (0-100)
    - Root cause analysis
    - Alert threshold configuration
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        alert_thresholds: Dict[str, float] = None
    ):
        """
        Initialize the drift analyzer.
        
        Args:
            methods: List of methods to use
            alert_thresholds: Custom thresholds for alerts
        """
        self.methods = methods or ['psi', 'ks_test', 'js_divergence', 'chi_square']
        default_thresholds = {
            'psi': 0.2,
            'ks_pvalue': 0.05,
            'js_divergence': 0.1,
            'chi_square_pvalue': 0.05
        }
        # Merge provided thresholds with defaults
        self.alert_thresholds = {**default_thresholds, **(alert_thresholds or {})}
        self.results_ = None
        
    def analyze(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = None
    ) -> Dict[str, Any]:
        """
        Analyze drift between baseline and current data.
        
        Args:
            baseline_data: Reference/training data
            current_data: Current/production data
            target_column: Optional target for concept drift
            
        Returns:
            Dictionary with drift analysis results
        """
        results = {
            'summary': {},
            'feature_drift': {},
            'overall_drift_score': 0.0,
            'drifted_features': [],
            'alerts': []
        }
        
        # Get common columns
        common_cols = list(set(baseline_data.columns) & set(current_data.columns))
        if target_column:
            common_cols = [c for c in common_cols if c != target_column]
        
        numeric_cols = baseline_data[common_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = baseline_data[common_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analyze each feature
        drift_scores = []
        
        for col in numeric_cols:
            feature_result = self._analyze_numeric_feature(
                baseline_data[col].dropna(),
                current_data[col].dropna()
            )
            feature_result['feature_type'] = 'numeric'
            results['feature_drift'][col] = feature_result
            drift_scores.append(feature_result['drift_score'])
            
            if feature_result['drift_detected']:
                results['drifted_features'].append(col)
                results['alerts'].append({
                    'feature': col,
                    'severity': feature_result['severity'],
                    'message': f"Drift detected in {col}: {feature_result['primary_indicator']}"
                })
        
        for col in categorical_cols:
            feature_result = self._analyze_categorical_feature(
                baseline_data[col].dropna(),
                current_data[col].dropna()
            )
            feature_result['feature_type'] = 'categorical'
            results['feature_drift'][col] = feature_result
            drift_scores.append(feature_result['drift_score'])
            
            if feature_result['drift_detected']:
                results['drifted_features'].append(col)
                results['alerts'].append({
                    'feature': col,
                    'severity': feature_result['severity'],
                    'message': f"Drift detected in {col}: {feature_result['primary_indicator']}"
                })
        
        # Concept drift analysis
        if target_column and target_column in baseline_data.columns and target_column in current_data.columns:
            concept_drift = self._analyze_concept_drift(
                baseline_data, current_data, target_column
            )
            results['concept_drift'] = concept_drift
        
        # Overall summary
        results['overall_drift_score'] = float(np.mean(drift_scores)) if drift_scores else 0.0
        results['summary'] = {
            'total_features_analyzed': len(common_cols),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'features_with_drift': len(results['drifted_features']),
            'drift_percentage': len(results['drifted_features']) / len(common_cols) * 100 if common_cols else 0,
            'overall_drift_score': results['overall_drift_score'],
            'overall_severity': self._score_to_severity(results['overall_drift_score'])
        }
        
        self.results_ = results
        return results
    
    def _analyze_numeric_feature(
        self,
        baseline: pd.Series,
        current: pd.Series
    ) -> Dict[str, Any]:
        """Analyze drift for a numeric feature"""
        result = {
            'drift_detected': False,
            'drift_score': 0.0,
            'severity': 'none',
            'metrics': {},
            'primary_indicator': None
        }
        
        # PSI
        if 'psi' in self.methods:
            psi = self._calculate_psi(baseline, current)
            result['metrics']['psi'] = psi
            if psi > self.alert_thresholds['psi']:
                result['drift_detected'] = True
                result['primary_indicator'] = f"PSI={psi:.3f}"
        
        # KS Test
        if 'ks_test' in self.methods:
            ks_stat, ks_pvalue = stats.ks_2samp(baseline, current)
            result['metrics']['ks_statistic'] = float(ks_stat)
            result['metrics']['ks_pvalue'] = float(ks_pvalue)
            if ks_pvalue < self.alert_thresholds['ks_pvalue']:
                result['drift_detected'] = True
                if result['primary_indicator'] is None:
                    result['primary_indicator'] = f"KS p-value={ks_pvalue:.4f}"
        
        # Jensen-Shannon Divergence
        if 'js_divergence' in self.methods:
            js_div = self._calculate_js_divergence(baseline, current)
            result['metrics']['js_divergence'] = js_div
            if js_div > self.alert_thresholds['js_divergence']:
                result['drift_detected'] = True
                if result['primary_indicator'] is None:
                    result['primary_indicator'] = f"JS Divergence={js_div:.3f}"
        
        # Distribution statistics comparison
        result['statistics'] = {
            'baseline': {
                'mean': float(baseline.mean()),
                'std': float(baseline.std()),
                'median': float(baseline.median()),
                'min': float(baseline.min()),
                'max': float(baseline.max())
            },
            'current': {
                'mean': float(current.mean()),
                'std': float(current.std()),
                'median': float(current.median()),
                'min': float(current.min()),
                'max': float(current.max())
            }
        }
        
        # Calculate drift score (0-100)
        result['drift_score'] = self._calculate_drift_score(result['metrics'])
        result['severity'] = self._score_to_severity(result['drift_score'])
        
        return result
    
    def _analyze_categorical_feature(
        self,
        baseline: pd.Series,
        current: pd.Series
    ) -> Dict[str, Any]:
        """Analyze drift for a categorical feature"""
        result = {
            'drift_detected': False,
            'drift_score': 0.0,
            'severity': 'none',
            'metrics': {},
            'primary_indicator': None
        }
        
        # Get value distributions
        baseline_dist = baseline.value_counts(normalize=True)
        current_dist = current.value_counts(normalize=True)
        
        # Align distributions
        all_categories = set(baseline_dist.index) | set(current_dist.index)
        baseline_aligned = pd.Series(
            {cat: baseline_dist.get(cat, 0) for cat in all_categories}
        )
        current_aligned = pd.Series(
            {cat: current_dist.get(cat, 0) for cat in all_categories}
        )
        
        # Chi-Square Test
        if 'chi_square' in self.methods:
            # Create contingency table approximation
            baseline_counts = (baseline_aligned * len(baseline)).astype(int)
            current_counts = (current_aligned * len(current)).astype(int)
            
            # Ensure no zero expected values
            baseline_counts = baseline_counts + 1
            current_counts = current_counts + 1
            
            try:
                chi2, pvalue = stats.chisquare(current_counts, baseline_counts)
                result['metrics']['chi_square'] = float(chi2)
                result['metrics']['chi_square_pvalue'] = float(pvalue)
                
                if pvalue < self.alert_thresholds['chi_square_pvalue']:
                    result['drift_detected'] = True
                    result['primary_indicator'] = f"Chi-square p-value={pvalue:.4f}"
            except Exception:
                pass
        
        # JS Divergence for categorical
        if 'js_divergence' in self.methods:
            # Add small epsilon to avoid zeros
            p = baseline_aligned.values + 1e-10
            q = current_aligned.values + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            
            js_div = float(jensenshannon(p, q))
            result['metrics']['js_divergence'] = js_div
            
            if js_div > self.alert_thresholds['js_divergence']:
                result['drift_detected'] = True
                if result['primary_indicator'] is None:
                    result['primary_indicator'] = f"JS Divergence={js_div:.3f}"
        
        # New categories detection
        new_categories = set(current_dist.index) - set(baseline_dist.index)
        missing_categories = set(baseline_dist.index) - set(current_dist.index)
        
        result['category_changes'] = {
            'new_categories': list(new_categories),
            'missing_categories': list(missing_categories),
            'new_categories_count': len(new_categories),
            'missing_categories_count': len(missing_categories)
        }
        
        if len(new_categories) > 0:
            result['drift_detected'] = True
            if result['primary_indicator'] is None:
                result['primary_indicator'] = f"{len(new_categories)} new categories detected"
        
        # Distribution comparison
        result['distribution'] = {
            'baseline': baseline_dist.head(20).to_dict(),
            'current': current_dist.head(20).to_dict()
        }
        
        # Calculate drift score
        result['drift_score'] = self._calculate_drift_score(result['metrics'])
        result['severity'] = self._score_to_severity(result['drift_score'])
        
        return result
    
    def _calculate_psi(
        self,
        baseline: pd.Series,
        current: pd.Series,
        n_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index"""
        # Create bins from baseline
        try:
            bins = pd.qcut(baseline, q=n_bins, duplicates='drop', retbins=True)[1]
        except ValueError:
            bins = pd.cut(baseline, bins=n_bins, retbins=True)[1]
        
        # Ensure bins cover current data range
        bins[0] = min(bins[0], current.min()) - 0.001
        bins[-1] = max(bins[-1], current.max()) + 0.001
        
        # Calculate proportions
        baseline_counts = pd.cut(baseline, bins=bins).value_counts(normalize=True).sort_index()
        current_counts = pd.cut(current, bins=bins).value_counts(normalize=True).sort_index()
        
        # Align
        baseline_props = baseline_counts.values + 1e-10
        current_props = current_counts.values + 1e-10
        
        # Calculate PSI
        psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))
        
        return float(psi)
    
    def _calculate_js_divergence(
        self,
        baseline: pd.Series,
        current: pd.Series,
        n_bins: int = 50
    ) -> float:
        """Calculate Jensen-Shannon Divergence"""
        # Create histograms
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
        current_hist, _ = np.histogram(current, bins=bins, density=True)
        
        # Normalize and add small epsilon
        baseline_hist = baseline_hist + 1e-10
        current_hist = current_hist + 1e-10
        baseline_hist = baseline_hist / baseline_hist.sum()
        current_hist = current_hist / current_hist.sum()
        
        return float(jensenshannon(baseline_hist, current_hist))
    
    def _calculate_drift_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall drift score (0-100)"""
        scores = []
        
        if 'psi' in metrics:
            # PSI: 0-0.1 (no drift), 0.1-0.2 (slight), 0.2+ (significant)
            psi_score = min(metrics['psi'] / 0.25 * 100, 100)
            scores.append(psi_score)
        
        if 'ks_pvalue' in metrics:
            # Lower p-value = higher drift
            ks_score = (1 - metrics['ks_pvalue']) * 100
            scores.append(ks_score)
        
        if 'js_divergence' in metrics:
            # JS: 0-1 scale
            js_score = min(metrics['js_divergence'] / 0.5 * 100, 100)
            scores.append(js_score)
        
        if 'chi_square_pvalue' in metrics:
            chi_score = (1 - metrics['chi_square_pvalue']) * 100
            scores.append(chi_score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _score_to_severity(self, score: float) -> str:
        """Convert drift score to severity level"""
        if score < 10:
            return 'none'
        elif score < 30:
            return 'low'
        elif score < 50:
            return 'medium'
        elif score < 70:
            return 'high'
        else:
            return 'critical'
    
    def _analyze_concept_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze concept drift (changes in target relationship)"""
        result = {
            'drift_detected': False,
            'severity': 'none',
            'metrics': {}
        }
        
        baseline_target = baseline_data[target_column]
        current_target = current_data[target_column]
        
        # Target distribution drift
        if baseline_target.dtype in ['int64', 'float64']:
            # Regression target
            ks_stat, ks_pvalue = stats.ks_2samp(
                baseline_target.dropna(),
                current_target.dropna()
            )
            result['metrics']['target_ks_statistic'] = float(ks_stat)
            result['metrics']['target_ks_pvalue'] = float(ks_pvalue)
            
            if ks_pvalue < 0.05:
                result['drift_detected'] = True
                result['severity'] = 'high'
        else:
            # Classification target
            baseline_dist = baseline_target.value_counts(normalize=True)
            current_dist = current_target.value_counts(normalize=True)
            
            result['metrics']['baseline_class_distribution'] = baseline_dist.to_dict()
            result['metrics']['current_class_distribution'] = current_dist.to_dict()
            
            # Check for class distribution shift
            for cls in set(baseline_dist.index) | set(current_dist.index):
                baseline_prop = baseline_dist.get(cls, 0)
                current_prop = current_dist.get(cls, 0)
                
                if abs(baseline_prop - current_prop) > 0.1:
                    result['drift_detected'] = True
                    result['severity'] = 'high'
                    break
        
        return result
    
    def generate_report(self) -> str:
        """Generate a human-readable drift report"""
        if self.results_ is None:
            return "No analysis results available. Run analyze() first."
        
        report = []
        report.append("=" * 60)
        report.append("DATA DRIFT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        summary = self.results_['summary']
        report.append(f"Features Analyzed: {summary['total_features_analyzed']}")
        report.append(f"  - Numeric: {summary['numeric_features']}")
        report.append(f"  - Categorical: {summary['categorical_features']}")
        report.append(f"Features with Drift: {summary['features_with_drift']} ({summary['drift_percentage']:.1f}%)")
        report.append(f"Overall Drift Score: {summary['overall_drift_score']:.1f}/100")
        report.append(f"Overall Severity: {summary['overall_severity'].upper()}")
        report.append("")
        
        if self.results_['alerts']:
            report.append("-" * 60)
            report.append("ALERTS")
            report.append("-" * 60)
            for alert in self.results_['alerts']:
                report.append(f"⚠️  [{alert['severity'].upper()}] {alert['feature']}: {alert['message']}")
            report.append("")
        
        report.append("-" * 60)
        report.append("FEATURE DETAILS")
        report.append("-" * 60)
        
        for feature, details in self.results_['feature_drift'].items():
            status = "🔴 DRIFT" if details['drift_detected'] else "🟢 OK"
            report.append(f"\n{feature} ({details['feature_type']}): {status}")
            report.append(f"  Drift Score: {details['drift_score']:.1f}/100")
            report.append(f"  Severity: {details['severity']}")
            
            if details['metrics']:
                report.append("  Metrics:")
                for metric, value in details['metrics'].items():
                    if isinstance(value, float):
                        report.append(f"    - {metric}: {value:.4f}")
        
        return "\n".join(report)

