"""
Pipeline package for DataAiPrep.

Contains:
- setup: DataAiPrepSetup entry point (PyCaret-like setup)
- feature_engineering: Automated feature creation utilities
- transformations: Scaling, normalization, and distribution transforms
- quality_scoring: Multi-dimensional data quality scoring
- exporter: PipelineExporter for serialization and metadata
- advisor: IntelligentAdvisor recommendation engine
- quickml: QuickML for quick model comparison on preprocessed data
- experiment: ExperimentTracker for experiment logging
"""

from .setup import DataAiPrepSetup  # noqa: F401
from .exporter import PipelineExporter  # noqa: F401
from .advisor import IntelligentAdvisor  # noqa: F401
from .quickml import QuickML  # noqa: F401
from .experiment import ExperimentTracker  # noqa: F401



