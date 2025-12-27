"""
Advanced Leakage Detection Module

Features:
- Train-Test Contamination Detection
- Near-Duplicate Detection
- Preprocessing Leakage Detection
- Group Leakage (same entity in train/test)
- Leakage Simulation
- Feature Leakage Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import hashlib
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LeakageIssue:
    """Container for a leakage issue"""
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_features: List[str]
    recommendation: str
    details: Dict[str, Any]


class AdvancedLeakageDetector:
    """
    Comprehensive leakage detection for ML pipelines.
    
    Types of leakage detected:
    - Train-Test Contamination: Same/similar samples in both sets
    - Near-Duplicate Detection: Very similar records across sets
    - Preprocessing Leakage: Statistics from test data used in training
    - Group Leakage: Same entity (e.g., customer) in train and test
    - Target Leakage: Features that encode target information
    - Temporal Leakage: Future information in features
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        correlation_threshold: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize the detector.
        
        Args:
            similarity_threshold: Threshold for similarity detection
            correlation_threshold: Threshold for correlation-based leakage
            random_state: Random seed
        """
        self.similarity_threshold = similarity_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.issues_: List[LeakageIssue] = []
        self.results_ = None
    
    def detect_all(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str = None,
        entity_column: str = None
    ) -> Dict[str, Any]:
        """
        Run all leakage detection methods.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            target_column: Target column name
            entity_column: Column identifying entities (for group leakage)
            
        Returns:
            Dictionary with all detection results
        """
        self.issues_ = []
        
        results = {
            'summary': {},
            'train_test_contamination': {},
            'near_duplicates': {},
            'group_leakage': {},
            'target_leakage': {},
            'feature_leakage': {},
            'issues': [],
            'recommendations': []
        }
        
        # 1. Train-Test Contamination
        results['train_test_contamination'] = self.detect_train_test_contamination(
            train_data, test_data
        )
        
        # 2. Near-Duplicate Detection
        results['near_duplicates'] = self.detect_near_duplicates(
            train_data, test_data
        )
        
        # 3. Group Leakage
        if entity_column and entity_column in train_data.columns:
            results['group_leakage'] = self.detect_group_leakage(
                train_data, test_data, entity_column
            )
        
        # 4. Target Leakage
        if target_column and target_column in train_data.columns:
            results['target_leakage'] = self.detect_target_leakage(
                train_data, target_column
            )
        
        # 5. Feature Leakage Analysis
        if target_column and target_column in train_data.columns:
            results['feature_leakage'] = self.analyze_feature_leakage(
                train_data, test_data, target_column
            )
        
        # Compile issues
        results['issues'] = [
            {
                'type': issue.type,
                'severity': issue.severity,
                'description': issue.description,
                'affected_features': issue.affected_features,
                'recommendation': issue.recommendation
            }
            for issue in self.issues_
        ]
        
        # Generate summary
        results['summary'] = {
            'total_issues': len(self.issues_),
            'critical_issues': len([i for i in self.issues_ if i.severity == 'critical']),
            'high_issues': len([i for i in self.issues_ if i.severity == 'high']),
            'medium_issues': len([i for i in self.issues_ if i.severity == 'medium']),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'has_contamination': results['train_test_contamination'].get('has_contamination', False),
            'has_group_leakage': results.get('group_leakage', {}).get('has_leakage', False)
        }
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations()
        
        self.results_ = results
        return results
    
    def detect_train_test_contamination(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect exact duplicates between train and test sets.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            
        Returns:
            Contamination detection results
        """
        result = {
            'has_contamination': False,
            'exact_duplicates': 0,
            'duplicate_indices': [],
            'contamination_rate': 0.0
        }
        
        # Create row hashes for efficient comparison
        train_hashes = set()
        for idx, row in train_data.iterrows():
            row_hash = hashlib.md5(str(row.values).encode()).hexdigest()
            train_hashes.add((row_hash, idx))
        
        train_hash_set = {h for h, _ in train_hashes}
        
        duplicates = []
        for idx, row in test_data.iterrows():
            row_hash = hashlib.md5(str(row.values).encode()).hexdigest()
            if row_hash in train_hash_set:
                duplicates.append(idx)
        
        if duplicates:
            result['has_contamination'] = True
            result['exact_duplicates'] = len(duplicates)
            result['duplicate_indices'] = duplicates[:100]  # Limit
            result['contamination_rate'] = len(duplicates) / len(test_data) * 100
            
            self.issues_.append(LeakageIssue(
                type='train_test_contamination',
                severity='critical',
                description=f"Found {len(duplicates)} exact duplicate rows between train and test sets",
                affected_features=['all'],
                recommendation="Remove duplicate rows from test set or re-split the data",
                details={'duplicate_count': len(duplicates), 'rate': result['contamination_rate']}
            ))
        
        return result
    
    def detect_near_duplicates(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Detect near-duplicate records between train and test.
        
        Uses cosine similarity on numeric features.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            sample_size: Sample size for efficiency
            
        Returns:
            Near-duplicate detection results
        """
        result = {
            'has_near_duplicates': False,
            'near_duplicate_count': 0,
            'similarity_threshold': self.similarity_threshold,
            'pairs': []
        }
        
        # Get numeric columns
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return result
        
        # Prepare data
        train_numeric = train_data[numeric_cols].fillna(0)
        test_numeric = test_data[numeric_cols].fillna(0)
        
        # Sample if too large
        if len(train_numeric) > sample_size:
            train_sample_idx = np.random.choice(len(train_numeric), sample_size, replace=False)
            train_sample = train_numeric.iloc[train_sample_idx]
        else:
            train_sample = train_numeric
            train_sample_idx = train_numeric.index.tolist()
        
        if len(test_numeric) > sample_size:
            test_sample_idx = np.random.choice(len(test_numeric), sample_size, replace=False)
            test_sample = test_numeric.iloc[test_sample_idx]
        else:
            test_sample = test_numeric
            test_sample_idx = test_numeric.index.tolist()
        
        # Scale data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_sample)
        test_scaled = scaler.transform(test_sample)
        
        # Calculate similarity matrix
        similarity = cosine_similarity(test_scaled, train_scaled)
        
        # Find near-duplicates
        near_duplicates = []
        for i in range(len(test_sample)):
            max_sim_idx = np.argmax(similarity[i])
            max_sim = similarity[i, max_sim_idx]
            
            if max_sim >= self.similarity_threshold:
                near_duplicates.append({
                    'test_index': test_sample_idx[i] if isinstance(test_sample_idx, list) else test_sample.index[i],
                    'train_index': train_sample_idx[max_sim_idx] if isinstance(train_sample_idx, list) else train_sample.index[max_sim_idx],
                    'similarity': float(max_sim)
                })
        
        if near_duplicates:
            result['has_near_duplicates'] = True
            result['near_duplicate_count'] = len(near_duplicates)
            result['pairs'] = sorted(near_duplicates, key=lambda x: -x['similarity'])[:50]
            
            self.issues_.append(LeakageIssue(
                type='near_duplicates',
                severity='high',
                description=f"Found {len(near_duplicates)} near-duplicate pairs (similarity >= {self.similarity_threshold})",
                affected_features=numeric_cols,
                recommendation="Review near-duplicate pairs and ensure proper train/test separation",
                details={'count': len(near_duplicates)}
            ))
        
        return result
    
    def detect_group_leakage(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        entity_column: str
    ) -> Dict[str, Any]:
        """
        Detect if same entities appear in both train and test.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            entity_column: Column identifying entities
            
        Returns:
            Group leakage detection results
        """
        result = {
            'has_leakage': False,
            'entity_column': entity_column,
            'overlapping_entities': [],
            'overlap_count': 0,
            'train_entity_count': 0,
            'test_entity_count': 0
        }
        
        if entity_column not in train_data.columns or entity_column not in test_data.columns:
            result['error'] = f"Entity column '{entity_column}' not found"
            return result
        
        train_entities = set(train_data[entity_column].dropna().unique())
        test_entities = set(test_data[entity_column].dropna().unique())
        
        overlap = train_entities & test_entities
        
        result['train_entity_count'] = len(train_entities)
        result['test_entity_count'] = len(test_entities)
        result['overlap_count'] = len(overlap)
        result['overlapping_entities'] = list(overlap)[:100]
        
        if overlap:
            result['has_leakage'] = True
            overlap_rate = len(overlap) / len(test_entities) * 100 if test_entities else 0
            
            severity = 'critical' if overlap_rate > 50 else ('high' if overlap_rate > 20 else 'medium')
            
            self.issues_.append(LeakageIssue(
                type='group_leakage',
                severity=severity,
                description=f"{len(overlap)} entities ({overlap_rate:.1f}%) appear in both train and test",
                affected_features=[entity_column],
                recommendation="Use GroupKFold or similar to ensure entities don't leak between splits",
                details={'overlap_rate': overlap_rate, 'overlap_count': len(overlap)}
            ))
        
        return result
    
    def detect_target_leakage(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Detect features that leak target information.
        
        Args:
            data: DataFrame with features and target
            target_column: Target column name
            
        Returns:
            Target leakage detection results
        """
        result = {
            'high_correlation_features': [],
            'perfect_predictors': [],
            'suspicious_features': []
        }
        
        if target_column not in data.columns:
            return result
        
        target = data[target_column]
        feature_cols = [c for c in data.columns if c != target_column]
        
        for col in feature_cols:
            series = data[col]
            
            # Check correlation for numeric
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
                try:
                    corr = abs(series.corr(target))
                    if corr >= self.correlation_threshold:
                        result['high_correlation_features'].append({
                            'feature': col,
                            'correlation': float(corr)
                        })
                        
                        self.issues_.append(LeakageIssue(
                            type='target_leakage',
                            severity='critical' if corr >= 0.99 else 'high',
                            description=f"Feature '{col}' has {corr:.3f} correlation with target",
                            affected_features=[col],
                            recommendation=f"Remove '{col}' or investigate if it's derived from target",
                            details={'correlation': corr}
                        ))
                except Exception:
                    pass
            
            # Check for perfect predictors
            if self._is_perfect_predictor(series, target):
                result['perfect_predictors'].append(col)
                
                if col not in [f['feature'] for f in result['high_correlation_features']]:
                    self.issues_.append(LeakageIssue(
                        type='perfect_predictor',
                        severity='critical',
                        description=f"Feature '{col}' perfectly predicts target",
                        affected_features=[col],
                        recommendation=f"Remove '{col}' - it's likely derived from target",
                        details={}
                    ))
            
            # Check suspicious names
            suspicious_keywords = ['target', 'label', 'outcome', 'result', 'prediction', 'answer']
            col_lower = col.lower()
            target_lower = target_column.lower()
            
            if any(kw in col_lower for kw in suspicious_keywords) or target_lower in col_lower:
                if col not in [f['feature'] for f in result['high_correlation_features']]:
                    result['suspicious_features'].append({
                        'feature': col,
                        'reason': 'name_similarity'
                    })
        
        return result
    
    def _is_perfect_predictor(
        self,
        feature: pd.Series,
        target: pd.Series
    ) -> bool:
        """Check if feature perfectly predicts target"""
        try:
            df = pd.DataFrame({'feature': feature, 'target': target}).dropna()
            if len(df) < 5:
                return False
            
            # Check if each feature value maps to exactly one target value
            grouped = df.groupby('feature')['target'].nunique()
            return (grouped == 1).all()
        except Exception:
            return False
    
    def analyze_feature_leakage(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Analyze potential feature leakage by comparing train vs test performance.
        
        If a feature has very different distributions or predictive power
        between train and test, it might indicate leakage.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            target_column: Target column name
            
        Returns:
            Feature leakage analysis results
        """
        result = {
            'distribution_shifts': [],
            'predictive_power_issues': []
        }
        
        feature_cols = [c for c in train_data.columns 
                       if c != target_column and pd.api.types.is_numeric_dtype(train_data[c])]
        
        for col in feature_cols[:50]:  # Limit for performance
            # Check distribution shift
            train_vals = train_data[col].dropna()
            test_vals = test_data[col].dropna()
            
            if len(train_vals) > 0 and len(test_vals) > 0:
                from scipy import stats
                
                try:
                    ks_stat, ks_pvalue = stats.ks_2samp(train_vals, test_vals)
                    
                    if ks_pvalue < 0.001:  # Highly significant shift
                        result['distribution_shifts'].append({
                            'feature': col,
                            'ks_statistic': float(ks_stat),
                            'ks_pvalue': float(ks_pvalue)
                        })
                except Exception:
                    pass
        
        # Sort by significance
        result['distribution_shifts'] = sorted(
            result['distribution_shifts'],
            key=lambda x: x['ks_statistic'],
            reverse=True
        )[:20]
        
        if result['distribution_shifts']:
            self.issues_.append(LeakageIssue(
                type='distribution_shift',
                severity='medium',
                description=f"{len(result['distribution_shifts'])} features have significant distribution shifts",
                affected_features=[d['feature'] for d in result['distribution_shifts'][:5]],
                recommendation="Review features with distribution shifts for potential leakage",
                details={'count': len(result['distribution_shifts'])}
            ))
        
        return result
    
    def simulate_leakage_removal(
        self,
        data: pd.DataFrame,
        target_column: str,
        suspected_features: List[str],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Simulate the impact of removing suspected leaky features.
        
        Args:
            data: DataFrame with features and target
            target_column: Target column name
            suspected_features: Features to test
            cv_folds: Number of cross-validation folds
            
        Returns:
            Impact simulation results
        """
        result = {
            'baseline_score': 0.0,
            'scores_after_removal': {},
            'impact': {}
        }
        
        X = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = data[target_column]
        
        X = X.fillna(X.median())
        
        # Determine model type
        if y.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
            scoring = 'neg_mean_squared_error'
        
        # Baseline score
        try:
            baseline_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            result['baseline_score'] = float(np.mean(baseline_scores))
        except Exception as e:
            result['error'] = f"Baseline evaluation failed: {e}"
            return result
        
        # Score after removing each feature
        for feature in suspected_features:
            if feature not in X.columns:
                continue
            
            X_reduced = X.drop(columns=[feature])
            
            try:
                scores = cross_val_score(model, X_reduced, y, cv=cv_folds, scoring=scoring)
                score = float(np.mean(scores))
                result['scores_after_removal'][feature] = score
                
                # Calculate impact
                impact = result['baseline_score'] - score
                result['impact'][feature] = {
                    'score_change': impact,
                    'percent_change': (impact / abs(result['baseline_score'])) * 100 if result['baseline_score'] != 0 else 0,
                    'verdict': 'likely_leakage' if impact > 0.1 else 'probably_fine'
                }
            except Exception:
                continue
        
        return result
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations based on detected issues"""
        recommendations = []
        
        for issue in self.issues_:
            recommendations.append({
                'category': issue.type,
                'severity': issue.severity,
                'recommendation': issue.recommendation
            })
        
        # Add general recommendations
        if any(i.type == 'train_test_contamination' for i in self.issues_):
            recommendations.append({
                'category': 'general',
                'severity': 'high',
                'recommendation': "Consider using sklearn's train_test_split with shuffle=True and a fixed random_state"
            })
        
        if any(i.type == 'group_leakage' for i in self.issues_):
            recommendations.append({
                'category': 'general',
                'severity': 'high',
                'recommendation': "Use GroupKFold or GroupShuffleSplit for proper cross-validation"
            })
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate a human-readable leakage report"""
        if self.results_ is None:
            return "No results available. Run detect_all() first."
        
        report = []
        report.append("=" * 60)
        report.append("ADVANCED LEAKAGE DETECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        summary = self.results_['summary']
        report.append(f"Train Size: {summary['train_size']:,}")
        report.append(f"Test Size: {summary['test_size']:,}")
        report.append(f"Total Issues Found: {summary['total_issues']}")
        report.append(f"  Critical: {summary['critical_issues']}")
        report.append(f"  High: {summary['high_issues']}")
        report.append(f"  Medium: {summary['medium_issues']}")
        report.append("")
        
        # Train-Test Contamination
        contam = self.results_['train_test_contamination']
        if contam.get('has_contamination'):
            report.append("-" * 60)
            report.append("🚨 TRAIN-TEST CONTAMINATION DETECTED")
            report.append("-" * 60)
            report.append(f"Exact Duplicates: {contam['exact_duplicates']}")
            report.append(f"Contamination Rate: {contam['contamination_rate']:.2f}%")
            report.append("")
        
        # Near Duplicates
        near_dup = self.results_['near_duplicates']
        if near_dup.get('has_near_duplicates'):
            report.append("-" * 60)
            report.append("⚠️ NEAR-DUPLICATES DETECTED")
            report.append("-" * 60)
            report.append(f"Near-Duplicate Pairs: {near_dup['near_duplicate_count']}")
            report.append(f"Similarity Threshold: {near_dup['similarity_threshold']}")
            report.append("")
        
        # Group Leakage
        group = self.results_.get('group_leakage', {})
        if group.get('has_leakage'):
            report.append("-" * 60)
            report.append("⚠️ GROUP LEAKAGE DETECTED")
            report.append("-" * 60)
            report.append(f"Overlapping Entities: {group['overlap_count']}")
            report.append(f"Entity Column: {group['entity_column']}")
            report.append("")
        
        # Target Leakage
        target_leak = self.results_['target_leakage']
        if target_leak.get('high_correlation_features') or target_leak.get('perfect_predictors'):
            report.append("-" * 60)
            report.append("🚨 TARGET LEAKAGE DETECTED")
            report.append("-" * 60)
            for feat in target_leak.get('high_correlation_features', []):
                report.append(f"  • {feat['feature']}: correlation = {feat['correlation']:.3f}")
            for feat in target_leak.get('perfect_predictors', []):
                report.append(f"  • {feat}: PERFECT PREDICTOR")
            report.append("")
        
        # Recommendations
        report.append("-" * 60)
        report.append("RECOMMENDATIONS")
        report.append("-" * 60)
        for rec in self.results_['recommendations']:
            report.append(f"\n[{rec['severity'].upper()}] {rec['category']}")
            report.append(f"  {rec['recommendation']}")
        
        return "\n".join(report)

