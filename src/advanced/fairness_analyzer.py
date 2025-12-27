"""
Fairness & Bias Detection Module

Features:
- Pre-training bias detection
- Protected attribute analysis
- Class imbalance metrics
- Proxy detection
- Intersectional analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BiasMetric:
    """Container for bias metric result"""
    metric_name: str
    value: float
    threshold: float
    bias_detected: bool
    affected_group: str
    interpretation: str


class FairnessAnalyzer:
    """
    Comprehensive fairness and bias detection.
    
    Pre-Training Bias Metrics:
    - Class Imbalance (CI)
    - Difference in Proportions of Labels (DPL)
    - Kullback-Leibler Divergence
    - Jensen-Shannon Divergence
    - Conditional Demographic Disparity
    
    Protected Attribute Analysis:
    - Proxy detection
    - Intersectional analysis
    - Fairness impact simulation
    """
    
    def __init__(
        self,
        protected_columns: List[str] = None,
        ci_threshold: float = 0.2,
        dpl_threshold: float = 0.1
    ):
        """
        Initialize the fairness analyzer.
        
        Args:
            protected_columns: List of protected attribute columns
            ci_threshold: Threshold for class imbalance
            dpl_threshold: Threshold for label proportion difference
        """
        self.protected_columns = protected_columns or []
        self.ci_threshold = ci_threshold
        self.dpl_threshold = dpl_threshold
        self.results_ = None
        
    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fairness analysis.
        
        Args:
            data: DataFrame to analyze
            target_column: Name of target/label column
            protected_columns: Override protected columns
            
        Returns:
            Dictionary with fairness analysis results
        """
        if protected_columns:
            self.protected_columns = protected_columns
        
        results = {
            'summary': {},
            'class_imbalance': self._analyze_class_imbalance(data, target_column),
            'protected_attribute_analysis': {},
            'proxy_detection': {},
            'intersectional_analysis': {},
            'recommendations': []
        }
        
        # Analyze each protected attribute
        for protected_col in self.protected_columns:
            if protected_col in data.columns:
                results['protected_attribute_analysis'][protected_col] = \
                    self._analyze_protected_attribute(data, target_column, protected_col)
        
        # Proxy detection
        results['proxy_detection'] = self._detect_proxies(data, target_column)
        
        # Intersectional analysis (if multiple protected attributes)
        if len(self.protected_columns) >= 2:
            results['intersectional_analysis'] = self._intersectional_analysis(
                data, target_column
            )
        
        # Generate summary and recommendations
        results['summary'] = self._generate_summary(results)
        results['recommendations'] = self._generate_recommendations(results)
        
        self.results_ = results
        return results
    
    def _analyze_class_imbalance(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze class imbalance in the target variable"""
        target = data[target_column].dropna()
        value_counts = target.value_counts()
        
        # Class Imbalance (CI) metric
        total = len(target)
        minority_count = value_counts.min()
        majority_count = value_counts.max()
        
        ci = (majority_count - minority_count) / total
        imbalance_ratio = majority_count / minority_count if minority_count > 0 else np.inf
        
        return {
            'class_distribution': value_counts.to_dict(),
            'class_proportions': (value_counts / total).to_dict(),
            'class_imbalance_ci': float(ci),
            'imbalance_ratio': float(imbalance_ratio),
            'minority_class': value_counts.idxmin(),
            'majority_class': value_counts.idxmax(),
            'is_imbalanced': ci > self.ci_threshold,
            'severity': self._imbalance_severity(imbalance_ratio)
        }
    
    def _imbalance_severity(self, ratio: float) -> str:
        """Determine imbalance severity"""
        if ratio < 2:
            return 'none'
        elif ratio < 5:
            return 'mild'
        elif ratio < 10:
            return 'moderate'
        elif ratio < 50:
            return 'severe'
        else:
            return 'extreme'
    
    def _analyze_protected_attribute(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_column: str
    ) -> Dict[str, Any]:
        """Analyze bias related to a protected attribute"""
        result = {
            'attribute': protected_column,
            'groups': {},
            'metrics': {},
            'bias_detected': False
        }
        
        # Get groups
        groups = data[protected_column].dropna().unique()
        result['n_groups'] = len(groups)
        
        # Analyze each group
        group_stats = {}
        for group in groups:
            group_data = data[data[protected_column] == group]
            group_target = group_data[target_column].dropna()
            
            group_stats[str(group)] = {
                'count': len(group_data),
                'proportion': len(group_data) / len(data),
                'positive_rate': (group_target == group_target.mode().iloc[0]).mean() 
                    if len(group_target) > 0 else 0,
                'target_distribution': group_target.value_counts(normalize=True).to_dict()
            }
        
        result['groups'] = group_stats
        
        # Calculate bias metrics
        
        # 1. Difference in Proportions of Labels (DPL)
        if len(groups) >= 2:
            positive_rates = [
                group_stats[str(g)].get('positive_rate', 0) 
                for g in groups
            ]
            dpl = max(positive_rates) - min(positive_rates)
            result['metrics']['dpl'] = float(dpl)
            result['metrics']['dpl_bias_detected'] = dpl > self.dpl_threshold
        
        # 2. Statistical parity difference
        if len(groups) >= 2:
            result['metrics']['statistical_parity'] = self._calculate_statistical_parity(
                data, target_column, protected_column
            )
        
        # 3. Kullback-Leibler Divergence between groups
        result['metrics']['kl_divergence'] = self._calculate_group_kl_divergence(
            data, target_column, protected_column
        )
        
        # 4. Jensen-Shannon Divergence
        result['metrics']['js_divergence'] = self._calculate_group_js_divergence(
            data, target_column, protected_column
        )
        
        # Determine if bias is detected
        result['bias_detected'] = (
            result['metrics'].get('dpl_bias_detected', False) or
            result['metrics'].get('js_divergence', 0) > 0.1
        )
        
        return result
    
    def _calculate_statistical_parity(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_column: str
    ) -> Dict[str, Any]:
        """Calculate statistical parity difference"""
        # Get positive outcome rate by group
        rates = data.groupby(protected_column)[target_column].apply(
            lambda x: (x == x.mode().iloc[0]).mean() if len(x) > 0 else 0
        )
        
        return {
            'rates_by_group': rates.to_dict(),
            'max_difference': float(rates.max() - rates.min()),
            'disparate_impact_ratio': float(rates.min() / rates.max()) if rates.max() > 0 else 0
        }
    
    def _calculate_group_kl_divergence(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_column: str
    ) -> float:
        """Calculate KL divergence between group target distributions"""
        groups = data[protected_column].dropna().unique()
        
        if len(groups) < 2:
            return 0.0
        
        # Get distributions
        distributions = []
        for group in groups[:2]:  # Compare first two groups
            group_data = data[data[protected_column] == group][target_column]
            dist = group_data.value_counts(normalize=True)
            distributions.append(dist)
        
        # Align distributions
        all_values = set(distributions[0].index) | set(distributions[1].index)
        p = np.array([distributions[0].get(v, 1e-10) for v in all_values])
        q = np.array([distributions[1].get(v, 1e-10) for v in all_values])
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate KL divergence
        kl = np.sum(p * np.log(p / q + 1e-10))
        
        return float(kl)
    
    def _calculate_group_js_divergence(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_column: str
    ) -> float:
        """Calculate JS divergence between group target distributions"""
        groups = data[protected_column].dropna().unique()
        
        if len(groups) < 2:
            return 0.0
        
        # Get distributions
        distributions = []
        for group in groups[:2]:
            group_data = data[data[protected_column] == group][target_column]
            dist = group_data.value_counts(normalize=True)
            distributions.append(dist)
        
        # Align distributions
        all_values = set(distributions[0].index) | set(distributions[1].index)
        p = np.array([distributions[0].get(v, 1e-10) for v in all_values])
        q = np.array([distributions[1].get(v, 1e-10) for v in all_values])
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        return float(jensenshannon(p, q))
    
    def _detect_proxies(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Detect features that may be proxies for protected attributes"""
        result = {
            'detected_proxies': [],
            'correlation_matrix': {}
        }
        
        if not self.protected_columns:
            return result
        
        # Get numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_column]
        
        # Encode protected columns if categorical
        encoded_protected = {}
        for protected_col in self.protected_columns:
            if protected_col in data.columns:
                if data[protected_col].dtype == 'object':
                    le = LabelEncoder()
                    encoded_protected[protected_col] = le.fit_transform(
                        data[protected_col].astype(str)
                    )
                else:
                    encoded_protected[protected_col] = data[protected_col].values
        
        # Check correlations
        for feature in numeric_cols:
            feature_values = data[feature].fillna(data[feature].median()).values
            
            for protected_col, protected_values in encoded_protected.items():
                # Calculate correlation
                try:
                    corr, pvalue = stats.pearsonr(feature_values, protected_values)
                    
                    if abs(corr) > 0.5 and pvalue < 0.05:
                        result['detected_proxies'].append({
                            'feature': feature,
                            'protected_attribute': protected_col,
                            'correlation': float(corr),
                            'p_value': float(pvalue),
                            'proxy_risk': 'high' if abs(corr) > 0.7 else 'medium'
                        })
                        
                    result['correlation_matrix'][f"{feature}_vs_{protected_col}"] = float(corr)
                except Exception:
                    continue
        
        # Sort by correlation strength
        result['detected_proxies'] = sorted(
            result['detected_proxies'],
            key=lambda x: abs(x['correlation']),
            reverse=True
        )
        
        return result
    
    def _intersectional_analysis(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze bias across intersections of protected attributes"""
        result = {
            'intersections': {},
            'most_disadvantaged': None,
            'disparity_range': 0.0
        }
        
        if len(self.protected_columns) < 2:
            return result
        
        # Create intersection column
        data = data.copy()
        intersection_col = '_intersection_'
        
        # Combine first two protected attributes
        col1, col2 = self.protected_columns[:2]
        data[intersection_col] = (
            data[col1].astype(str) + '_' + data[col2].astype(str)
        )
        
        # Analyze each intersection
        positive_rates = {}
        for intersection in data[intersection_col].unique():
            group_data = data[data[intersection_col] == intersection]
            
            if len(group_data) >= 10:  # Minimum group size
                group_target = group_data[target_column].dropna()
                positive_rate = (group_target == group_target.mode().iloc[0]).mean() \
                    if len(group_target) > 0 else 0
                
                result['intersections'][intersection] = {
                    'count': len(group_data),
                    'proportion': len(group_data) / len(data),
                    'positive_rate': float(positive_rate)
                }
                positive_rates[intersection] = positive_rate
        
        # Find most disadvantaged intersection
        if positive_rates:
            min_rate_intersection = min(positive_rates, key=positive_rates.get)
            max_rate_intersection = max(positive_rates, key=positive_rates.get)
            
            result['most_disadvantaged'] = min_rate_intersection
            result['most_advantaged'] = max_rate_intersection
            result['disparity_range'] = float(
                positive_rates[max_rate_intersection] - 
                positive_rates[min_rate_intersection]
            )
        
        return result
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of fairness analysis"""
        summary = {
            'protected_attributes_analyzed': len(results['protected_attribute_analysis']),
            'bias_detected': False,
            'class_imbalanced': results['class_imbalance']['is_imbalanced'],
            'proxies_found': len(results['proxy_detection'].get('detected_proxies', [])),
            'overall_risk': 'low'
        }
        
        # Check for bias in protected attributes
        for attr, analysis in results['protected_attribute_analysis'].items():
            if analysis.get('bias_detected', False):
                summary['bias_detected'] = True
                break
        
        # Determine overall risk
        risk_factors = 0
        if summary['class_imbalanced']:
            risk_factors += 1
        if summary['bias_detected']:
            risk_factors += 2
        if summary['proxies_found'] > 0:
            risk_factors += 1
        
        if risk_factors >= 3:
            summary['overall_risk'] = 'high'
        elif risk_factors >= 2:
            summary['overall_risk'] = 'medium'
        else:
            summary['overall_risk'] = 'low'
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate fairness recommendations"""
        recommendations = []
        
        # Class imbalance recommendations
        if results['class_imbalance']['is_imbalanced']:
            severity = results['class_imbalance']['severity']
            recommendations.append({
                'category': 'Class Imbalance',
                'severity': severity,
                'issue': f"Target variable is imbalanced (ratio: {results['class_imbalance']['imbalance_ratio']:.1f}:1)",
                'recommendation': "Consider using SMOTE, class weights, or undersampling to address imbalance"
            })
        
        # Protected attribute bias
        for attr, analysis in results['protected_attribute_analysis'].items():
            if analysis.get('bias_detected', False):
                dpl = analysis['metrics'].get('dpl', 0)
                recommendations.append({
                    'category': 'Protected Attribute Bias',
                    'severity': 'high' if dpl > 0.2 else 'medium',
                    'issue': f"Bias detected in {attr} (DPL: {dpl:.3f})",
                    'recommendation': f"Review feature engineering and consider fairness constraints during training"
                })
        
        # Proxy detection
        for proxy in results['proxy_detection'].get('detected_proxies', [])[:3]:
            recommendations.append({
                'category': 'Proxy Variable',
                'severity': proxy['proxy_risk'],
                'issue': f"Feature '{proxy['feature']}' is correlated with {proxy['protected_attribute']} (r={proxy['correlation']:.2f})",
                'recommendation': "Consider removing or transforming this feature to reduce indirect discrimination"
            })
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate human-readable fairness report"""
        if self.results_ is None:
            return "No analysis results available. Run analyze() first."
        
        report = []
        report.append("=" * 60)
        report.append("FAIRNESS & BIAS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        summary = self.results_['summary']
        report.append(f"Overall Risk Level: {summary['overall_risk'].upper()}")
        report.append(f"Protected Attributes Analyzed: {summary['protected_attributes_analyzed']}")
        report.append(f"Class Imbalanced: {'Yes' if summary['class_imbalanced'] else 'No'}")
        report.append(f"Bias Detected: {'Yes' if summary['bias_detected'] else 'No'}")
        report.append(f"Proxy Variables Found: {summary['proxies_found']}")
        report.append("")
        
        # Class imbalance
        ci = self.results_['class_imbalance']
        report.append("-" * 60)
        report.append("CLASS IMBALANCE")
        report.append("-" * 60)
        report.append(f"Imbalance Ratio: {ci['imbalance_ratio']:.2f}:1")
        report.append(f"Majority Class: {ci['majority_class']}")
        report.append(f"Minority Class: {ci['minority_class']}")
        report.append(f"Severity: {ci['severity']}")
        report.append("")
        
        # Protected attribute analysis
        if self.results_['protected_attribute_analysis']:
            report.append("-" * 60)
            report.append("PROTECTED ATTRIBUTE ANALYSIS")
            report.append("-" * 60)
            
            for attr, analysis in self.results_['protected_attribute_analysis'].items():
                status = "⚠️ BIAS DETECTED" if analysis['bias_detected'] else "✓ No significant bias"
                report.append(f"\n{attr}: {status}")
                report.append(f"  Groups: {analysis['n_groups']}")
                
                if 'dpl' in analysis['metrics']:
                    report.append(f"  DPL: {analysis['metrics']['dpl']:.4f}")
                if 'js_divergence' in analysis['metrics']:
                    report.append(f"  JS Divergence: {analysis['metrics']['js_divergence']:.4f}")
        
        # Recommendations
        if self.results_['recommendations']:
            report.append("")
            report.append("-" * 60)
            report.append("RECOMMENDATIONS")
            report.append("-" * 60)
            
            for rec in self.results_['recommendations']:
                report.append(f"\n[{rec['severity'].upper()}] {rec['category']}")
                report.append(f"  Issue: {rec['issue']}")
                report.append(f"  Action: {rec['recommendation']}")
        
        return "\n".join(report)

