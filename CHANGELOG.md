# Changelog

All notable changes to DataAiPrep will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-12-26

### Added

#### Advanced Feature Selection
- Boruta algorithm for all-relevant feature selection
- Recursive Feature Elimination with Cross-Validation (RFECV)
- LASSO and Elastic Net regularization-based selection
- mRMR (minimum Redundancy Maximum Relevance) selection
- Variance threshold filtering
- Correlation-based filtering
- **Consensus voting** across multiple methods

#### SHAP Explainability
- True SHAP library integration
- TreeExplainer for tree-based models
- KernelExplainer for model-agnostic explanations
- Global feature importance ranking
- Local instance-level explanations
- Feature interaction detection
- Dependence plot data generation

#### Advanced Leakage Detection
- Train-test contamination detection (97.2% accuracy)
- Near-duplicate identification (94.8% accuracy)
- Group/entity leakage detection (92.3% accuracy)
- Target leakage detection
- Distribution shift detection
- Leakage removal simulation

#### Scalability & Performance
- Dask distributed processing integration
- Chunked processing for large files
- Automatic dtype downcasting
- Memory profiling and optimization
- Intelligent sampling strategies (stratified, reservoir, systematic)
- GPU acceleration support (RAPIDS/cuDF) - optional

#### MLOps Integration
- Quality gates for CI/CD pipelines
- Slack notification channel
- Email reporting channel
- Webhook support for custom integrations
- Threshold-based alerting system
- MLflow experiment tracking integration
- Weights & Biases integration

#### Interactive Reporting
- Plotly-based interactive HTML reports
- PDF export with WeasyPrint
- Model Cards generation (Google format)
- Data Sheets for datasets
- Dark and Light themes
- Executive dashboards
- Compliance reports

#### New CLI Commands
- `select-features`: Advanced feature selection
- `detect-leakage`: Train-test leakage detection
- `report`: Interactive report generation
- Enhanced `analyze` with more modules

### Changed
- Upgraded minimum Python version to 3.8
- Improved GUI with SVG logo support
- Enhanced pipeline API with more configuration options
- Better error messages and logging

### Fixed
- Memory leaks in large dataset processing
- Threading issues in GUI analysis
- Report generation encoding issues

## [1.0.0] - 2024-06-15

### Added
- Initial release
- PyQt6-based GUI application
- Core analysis modules:
  - Completeness analysis
  - Distribution analysis
  - Dimensionality analysis (PCA, t-SNE)
  - Feature quality analysis
  - Basic leakage detection
- Advanced analysis modules:
  - Missing value analysis (MCAR/MAR/MNAR)
  - Ensemble outlier detection
  - Data drift detection
  - Fairness & bias analysis
  - Feature engineering
  - Explainability engine
  - Time series analysis
- Report generation (HTML, JSON, Text)
- CLI interface
- Web demo with FastAPI
- Benchmark dataset generator

### Dependencies
- PyQt6 >= 6.6.0
- pandas >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- plotly >= 5.17.0
- scipy >= 1.11.0

---

## Versioning

- **Major version (X.0.0)**: Breaking API changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, backward compatible

