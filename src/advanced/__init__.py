"""
DataAiPrep Advanced Features Module

Expert-level data quality assessment and ML preprocessing capabilities
that go beyond PyCaret's functionality.

Modules:
- missing_value_analyzer: MCAR/MAR/MNAR classification, smart imputation
- outlier_detector: Multi-method ensemble outlier detection
- drift_detector: Data drift and model monitoring
- fairness_analyzer: Bias detection and fairness assessment
- feature_engineer: Advanced feature engineering
- explainability: SHAP integration and feature importance
- time_series_analyzer: Temporal data quality assessment
- pipeline: DataQualityPipeline for comprehensive analysis
- feature_selection: Boruta, RFE, LASSO, mRMR feature selection
- shap_explainer: True SHAP integration for model explainability
- scalability: Dask integration, memory optimization, chunked processing
- alerting: Slack, Email, Webhook notifications and quality gates
- enhanced_reporting: Interactive Plotly reports, PDF export, Model Cards
- advanced_leakage: Train-test contamination, group leakage detection
"""

from .pipeline import DataQualityPipeline
from .missing_value_analyzer import AdvancedMissingAnalyzer
from .outlier_detector import EnsembleOutlierDetector
from .drift_detector import DriftAnalyzer
from .fairness_analyzer import FairnessAnalyzer
from .feature_engineer import AdvancedFeatureEngineer
from .explainability import ExplainabilityEngine
from .time_series_analyzer import TimeSeriesAnalyzer

# New modules
from .feature_selection import AdvancedFeatureSelector
from .shap_explainer import SHAPExplainer
from .scalability import LargeDataProcessor, IntelligentSampler, MemoryProfiler
from .alerting import AlertManager, SlackChannel, EmailChannel, WebhookChannel, QualityGate
from .enhanced_reporting import InteractiveReportGenerator, ModelCardGenerator
from .advanced_leakage import AdvancedLeakageDetector

__all__ = [
    # Core modules
    'DataQualityPipeline',
    'AdvancedMissingAnalyzer',
    'EnsembleOutlierDetector',
    'DriftAnalyzer',
    'FairnessAnalyzer',
    'AdvancedFeatureEngineer',
    'ExplainabilityEngine',
    'TimeSeriesAnalyzer',
    
    # New feature selection
    'AdvancedFeatureSelector',
    
    # SHAP explainability
    'SHAPExplainer',
    
    # Scalability
    'LargeDataProcessor',
    'IntelligentSampler',
    'MemoryProfiler',
    
    # Alerting
    'AlertManager',
    'SlackChannel',
    'EmailChannel',
    'WebhookChannel',
    'QualityGate',
    
    # Enhanced reporting
    'InteractiveReportGenerator',
    'ModelCardGenerator',
    
    # Advanced leakage detection
    'AdvancedLeakageDetector'
]
