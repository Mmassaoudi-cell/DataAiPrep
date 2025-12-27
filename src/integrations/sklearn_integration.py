"""
Scikit-learn Integration for DataAiPrep

Provides sklearn-compatible transformers for data quality assessment
and preprocessing pipeline integration.

Usage:
    from src.integrations import DataAiPrepTransformer, create_sklearn_pipeline
    
    # Use in sklearn pipeline
    pipeline = create_sklearn_pipeline(
        quality_check=True,
        auto_preprocess=True,
        model=RandomForestClassifier()
    )
    pipeline.fit(X_train, y_train)
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class DataAiPrepTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for DataAiPrep quality assessment.
    
    Features:
    - Fits into sklearn Pipeline
    - Runs quality assessment during fit
    - Stores quality metrics for inspection
    - Optionally blocks fitting if quality issues detected
    """
    
    def __init__(
        self,
        target_column: Optional[str] = None,
        fail_on_issues: bool = False,
        missing_threshold: float = 0.20,
        leakage_threshold: float = 50.0,
        verbose: bool = True
    ):
        """
        Initialize DataAiPrep transformer.
        
        Args:
            target_column: Name of target column (will be excluded from X)
            fail_on_issues: Raise exception if critical issues found
            missing_threshold: Maximum allowed missing percentage (0-1)
            leakage_threshold: Maximum allowed leakage score (0-100)
            verbose: Print quality assessment results
        """
        self.target_column = target_column
        self.fail_on_issues = fail_on_issues
        self.missing_threshold = missing_threshold
        self.leakage_threshold = leakage_threshold
        self.verbose = verbose
        
        # Results storage
        self.quality_results_ = None
        self.issues_found_ = []
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """
        Fit the transformer (run quality assessment).
        
        Args:
            X: Input features (DataFrame or array)
            y: Target variable (optional)
            
        Returns:
            self
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Import analyzers
        from ..analysis.completeness_analyzer import CompletenessAnalyzer
        from ..analysis.distribution_analyzer import DistributionAnalyzer
        from ..analysis.leakage_detector import LeakageDetector
        
        # Run quality assessment
        self.quality_results_ = {
            'completeness': CompletenessAnalyzer().analyze(X),
            'distribution': DistributionAnalyzer().analyze(X, self.target_column),
            'leakage': LeakageDetector().analyze(X, self.target_column)
        }
        
        # Check for issues
        self._check_issues()
        
        if self.verbose:
            self._print_summary()
        
        if self.fail_on_issues and self.issues_found_:
            raise ValueError(
                f"Critical data quality issues found: {self.issues_found_}. "
                "Set fail_on_issues=False to proceed anyway."
            )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, y=None):
        """
        Transform the data (passthrough, assessment only).
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            X unchanged
        """
        return X
    
    def _check_issues(self):
        """Check quality results for critical issues"""
        self.issues_found_ = []
        
        completeness = self.quality_results_['completeness']
        leakage = self.quality_results_['leakage']
        
        # Check missing data
        missing_pct = completeness.get('overall_missing_percentage', 0) / 100
        if missing_pct > self.missing_threshold:
            self.issues_found_.append(
                f"High missing data: {missing_pct*100:.1f}% > {self.missing_threshold*100}%"
            )
        
        # Check leakage
        leakage_score = leakage.get('leakage_score', 0)
        if leakage_score > self.leakage_threshold:
            self.issues_found_.append(
                f"High leakage risk: {leakage_score:.1f} > {self.leakage_threshold}"
            )
        
        # Check for perfect correlations
        if len(leakage.get('perfect_correlations', [])) > 0:
            self.issues_found_.append(
                f"Perfect correlations detected: {len(leakage['perfect_correlations'])} pairs"
            )
    
    def _print_summary(self):
        """Print quality assessment summary"""
        completeness = self.quality_results_['completeness']
        leakage = self.quality_results_['leakage']
        
        print("\n" + "="*50)
        print("DataAiPrep Quality Assessment")
        print("="*50)
        print(f"Completeness Score: {completeness.get('completeness_score', 0):.1f}%")
        print(f"Missing Percentage: {completeness.get('overall_missing_percentage', 0):.1f}%")
        print(f"Leakage Risk Score: {leakage.get('leakage_score', 0):.1f}/100")
        print(f"Perfect Correlations: {len(leakage.get('perfect_correlations', []))}")
        
        if self.issues_found_:
            print("\n⚠️ Issues Found:")
            for issue in self.issues_found_:
                print(f"  - {issue}")
        else:
            print("\n✅ No critical issues found")
        print("="*50 + "\n")
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get the full quality assessment report"""
        if not self.is_fitted_:
            raise ValueError("Transformer has not been fitted yet")
        return self.quality_results_


class DataAiPrepPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible preprocessor using DataAiPrep's pipeline.
    
    Automatically:
    - Infers column types
    - Handles missing values
    - Scales numeric features
    - Encodes categorical features
    """
    
    def __init__(
        self,
        normalize: bool = True,
        normalize_method: str = 'zscore',
        encoding_method: str = 'auto',
        imputation_type: str = 'simple',
        numeric_imputation: str = 'mean',
        categorical_imputation: str = 'most_frequent'
    ):
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.encoding_method = encoding_method
        self.imputation_type = imputation_type
        self.numeric_imputation = numeric_imputation
        self.categorical_imputation = categorical_imputation
        
        self._setup = None
        self._pipeline = None
        self.feature_metadata_ = None
        
    def fit(self, X, y=None):
        """Fit the preprocessor"""
        from ..pipeline import DataAiPrepSetup
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        self._setup = DataAiPrepSetup()
        
        _, self._pipeline = self._setup.setup(
            data=X,
            target=None,
            normalize=self.normalize,
            normalize_method=self.normalize_method,
            encoding_method=self.encoding_method,
            imputation_type=self.imputation_type,
            numeric_imputation=self.numeric_imputation,
            categorical_imputation=self.categorical_imputation
        )
        
        self.feature_metadata_ = self._setup.feature_metadata_
        
        return self
    
    def transform(self, X, y=None):
        """Transform the data"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        return self._pipeline.transform(X)


def create_sklearn_pipeline(
    quality_check: bool = True,
    auto_preprocess: bool = True,
    model: Optional[BaseEstimator] = None,
    fail_on_quality_issues: bool = False,
    **preprocessor_kwargs
) -> Pipeline:
    """
    Create a complete sklearn pipeline with DataAiPrep integration.
    
    Args:
        quality_check: Include quality assessment step
        auto_preprocess: Include automatic preprocessing
        model: Final estimator (classifier or regressor)
        fail_on_quality_issues: Raise exception if quality issues found
        **preprocessor_kwargs: Additional arguments for DataAiPrepPreprocessor
        
    Returns:
        Sklearn Pipeline
    """
    steps = []
    
    if quality_check:
        steps.append((
            'quality_check',
            DataAiPrepTransformer(
                fail_on_issues=fail_on_quality_issues,
                verbose=True
            )
        ))
    
    if auto_preprocess:
        steps.append((
            'preprocess',
            DataAiPrepPreprocessor(**preprocessor_kwargs)
        ))
    
    if model is not None:
        steps.append(('model', model))
    
    return Pipeline(steps)


# Example usage code snippet
SKLEARN_EXAMPLE = '''
# Scikit-learn Pipeline Integration Example
# ==========================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from src.integrations import (
    DataAiPrepTransformer,
    DataAiPrepPreprocessor,
    create_sklearn_pipeline
)

# Load your data
df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Option 1: Use individual transformers
# -------------------------------------
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('quality_check', DataAiPrepTransformer(verbose=True)),
    ('preprocess', DataAiPrepPreprocessor(normalize=True)),
    ('model', RandomForestClassifier(n_estimators=100))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Option 2: Use the helper function
# ---------------------------------
pipeline = create_sklearn_pipeline(
    quality_check=True,
    auto_preprocess=True,
    model=RandomForestClassifier(n_estimators=100),
    fail_on_quality_issues=False,
    normalize=True,
    normalize_method='robust'
)

# Cross-validation with quality checks at each fold
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Option 3: Quality check only (no preprocessing)
# ------------------------------------------------
quality_checker = DataAiPrepTransformer(
    fail_on_issues=True,  # Raise exception if issues found
    missing_threshold=0.10,  # Max 10% missing allowed
    leakage_threshold=30.0  # Max leakage score of 30
)

try:
    quality_checker.fit(X)
    print("Data passed quality checks!")
except ValueError as e:
    print(f"Data quality issues: {e}")

# Access quality report
report = quality_checker.get_quality_report()
print(f"Completeness: {report['completeness']['completeness_score']:.1f}%")
'''

