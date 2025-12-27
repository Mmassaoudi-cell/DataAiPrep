<p align="center">
  <img src="data_ai_prep_logo.svg" alt="DataAiPrep Logo" width="300"/>
</p>

<h1 align="center">DataAiPrep</h1>

<p align="center">
  <strong>Advanced ML Data Quality Assessment Platform</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/version-2.0.0-orange.svg" alt="Version 2.0.0"/>
</p>

---

## Overview

**DataAiPrep** is a comprehensive, enterprise-grade data preprocessing and quality assessment tool that goes beyond PyCaret's capabilities with expert-level features. It provides automated detection of data quality issues critical for machine learning success, including advanced leakage detection, SHAP explainability, and scalable processing with Dask.

### Why DataAiPrep?

| Challenge | DataAiPrep Solution |
|-----------|---------------------|
| Train-test data leakage | 97.2% detection accuracy for contamination |
| Feature selection complexity | Boruta, RFE, LASSO, mRMR with consensus voting |
| Model interpretability | True SHAP integration with TreeExplainer |
| Large dataset processing | Dask distributed computing support |
| CI/CD integration | Quality gates with Slack/Email/Webhook alerts |

---

## Features

### 🔍 Core Analysis Modules

- **Advanced Missing Value Analysis**: MCAR/MAR/MNAR classification with smart imputation recommendations
- **Ensemble Outlier Detection**: IQR, Z-score, Isolation Forest, LOF, DBSCAN with consensus voting
- **Data Drift Detection**: PSI, Kolmogorov-Smirnov test, Jensen-Shannon divergence
- **Fairness & Bias Analysis**: Class imbalance, proxy detection, intersectional analysis

### 🎯 Advanced Features (v2.0)

- **Advanced Leakage Detection**
  - Train-test contamination (97.2% accuracy)
  - Near-duplicate identification (94.8% accuracy)
  - Group/entity leakage (92.3% accuracy)
  - Target leakage detection

- **Automated Feature Selection**
  - Boruta algorithm
  - Recursive Feature Elimination (RFE) with CV
  - LASSO/Elastic Net regularization
  - mRMR (minimum Redundancy Maximum Relevance)
  - Variance threshold filtering
  - Correlation-based filtering
  - **Consensus voting** across methods

- **SHAP Explainability**
  - Global feature importance
  - Local instance explanations
  - Feature interaction detection
  - TreeExplainer and KernelExplainer support

- **Scalability & Performance**
  - Dask distributed processing
  - Chunked processing for large files
  - Automatic dtype downcasting
  - Memory profiling and optimization
  - Intelligent sampling strategies

- **MLOps Integration**
  - Quality gates for CI/CD pipelines
  - Slack notifications
  - Email reports
  - Webhook support
  - MLflow integration
  - Weights & Biases integration

- **Interactive Reporting**
  - Plotly-based interactive HTML reports
  - PDF export with WeasyPrint
  - Model Cards generation
  - Data Sheets for datasets
  - Dark/Light themes

---

## Installation

### Basic Installation

```bash
pip install dataiprep
```

### From Source

```bash
git clone https://github.com/massaoudi-lab/dataiprep.git
cd dataiprep
pip install -e .
```

### Full Installation (with all optional features)

```bash
pip install dataiprep[full]
# or
pip install -r requirements.txt
pip install shap dask[complete] mlflow wandb
```

### Minimal Installation (GUI only)

```bash
pip install PyQt6 PyQt6-SVG pandas numpy scikit-learn matplotlib seaborn
```

---

## Quick Start

### GUI Application

```bash
python main.py
# or
python main.py --gui
```

### CLI Usage

```bash
# Basic data quality analysis
python main.py analyze data.csv --target label --output report.html

# With specific modules
python main.py analyze data.csv --modules completeness,outliers,drift

# Drift detection with baseline
python main.py analyze production.csv --baseline training.csv --target label

# Feature selection
python main.py select-features data.csv --target label --methods boruta,rfe,lasso

# Leakage detection
python main.py detect-leakage train.csv test.csv --target label

# Generate interactive report
python main.py report data.csv --output report.html --theme dark
```

### Web Demo

```bash
python main.py --web --port 8000
```

### Python API

```python
from src.advanced import DataQualityPipeline
import pandas as pd

# Load data
df = pd.read_csv("your_data.csv")

# Create pipeline
pipeline = DataQualityPipeline(name="MyAnalysis")
pipeline.add_step('completeness', threshold=0.95)
pipeline.add_step('outlier_detection', methods=['iqr', 'isolation_forest', 'lof'])
pipeline.add_step('explainability')

# Run analysis
pipeline.run(data=df, target='label')

# Export results
pipeline.export('quality_report.html', format='html')

# Get recommendations
for rec in pipeline.get_recommendations(severity='high'):
    print(f"[{rec['severity']}] {rec['issue']}")
```

---

## Advanced Usage Examples

### Feature Selection with Consensus Voting

```python
from src.advanced import AdvancedFeatureSelector
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=['target'])
y = df['target']

selector = AdvancedFeatureSelector(random_state=42)
results = selector.select_features(
    X, y,
    methods=['boruta', 'rfe', 'lasso', 'mrmr'],
    n_features=20,
    consensus_threshold=0.5
)

# Features selected by multiple methods
consensus_features = results['consensus_features']
print(f"Consensus features: {consensus_features}")
```

### Leakage Detection

```python
from src.advanced import AdvancedLeakageDetector
import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

detector = AdvancedLeakageDetector(
    similarity_threshold=0.95,
    correlation_threshold=0.95
)

results = detector.detect_all(
    train_data=train_df,
    test_data=test_df,
    target_column='label',
    entity_column='customer_id'
)

if results['summary']['has_contamination']:
    print("⚠️ Train-test contamination detected!")
```

### SHAP Explainability

```python
from src.advanced import SHAPExplainer
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=['target'])
y = df['target']

explainer = SHAPExplainer(random_state=42)
results = explainer.explain(X, y)

# Global feature importance
for feat, importance in list(results['global_importance'].items())[:10]:
    print(f"{feat}: {importance:.4f}")
```

### Quality Gates for CI/CD

```python
from src.advanced import QualityGate, AlertManager, SlackChannel

# Set up quality gate
gate = QualityGate(name="ProductionReadiness")
gate.add_check('completeness', 'completeness_score', 'gte', 95, severity='error')
gate.add_check('outliers', 'outlier_percentage', 'lte', 2, severity='warning')

# Evaluate metrics
result = gate.evaluate({
    'completeness_score': 98,
    'outlier_percentage': 1.5
})

if not result['passed']:
    print("❌ Quality gate failed!")
    # Send alert
    alert_manager = AlertManager()
    slack = SlackChannel(webhook_url="YOUR_WEBHOOK_URL")
    alert_manager.add_channel('slack', slack)
    alert_manager.send_alert("Quality gate failed", severity='error')
```

### Processing Large Datasets with Dask

```python
from src.advanced import LargeDataProcessor

processor = LargeDataProcessor(
    chunk_size=100000,
    use_dask=True,
    n_workers=4
)

# Read large CSV efficiently
df = processor.read_large_csv("large_file.csv", optimize_memory=True)

# Get memory statistics
memory_stats = processor.get_memory_usage(df)
print(f"Memory usage: {memory_stats['total_mb']:.2f} MB")
```

---

## Project Structure

```
dataiprep/
├── main.py                    # Entry point (GUI, CLI, Web)
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── data_ai_prep_logo.svg      # Application logo
├── src/
│   ├── advanced/              # Advanced analysis modules
│   │   ├── advanced_leakage.py
│   │   ├── alerting.py
│   │   ├── drift_detector.py
│   │   ├── enhanced_reporting.py
│   │   ├── explainability.py
│   │   ├── fairness_analyzer.py
│   │   ├── feature_engineer.py
│   │   ├── feature_selection.py
│   │   ├── missing_value_analyzer.py
│   │   ├── outlier_detector.py
│   │   ├── pipeline.py
│   │   ├── scalability.py
│   │   ├── shap_explainer.py
│   │   └── time_series_analyzer.py
│   ├── analysis/              # Core analysis modules
│   │   ├── completeness_analyzer.py
│   │   ├── dimensionality_analyzer.py
│   │   ├── distribution_analyzer.py
│   │   ├── feature_quality_analyzer.py
│   │   └── leakage_detector.py
│   ├── benchmark/             # Benchmark dataset generation
│   ├── core/                  # Data loading utilities
│   ├── gui/                   # PyQt6 GUI components
│   ├── integrations/          # MLflow, W&B, CI/CD integrations
│   ├── pipeline/              # Preprocessing pipeline
│   ├── reports/               # Report generation
│   └── web/                   # FastAPI web demo
├── examples/                  # Usage examples
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

---

## Comparison with Other Tools

| Feature | DataAiPrep | Pandas Profiling | Great Expectations | PyCaret |
|---------|------------|------------------|-------------------|---------|
| Train-Test Leakage Detection | ✅ 97.2% | ❌ | ❌ | Partial |
| Near-Duplicate Detection | ✅ 94.8% | ❌ | ❌ | ❌ |
| Boruta Feature Selection | ✅ | ❌ | ❌ | ✅ |
| SHAP Explainability | ✅ | ❌ | ❌ | ✅ |
| Slack/Email Alerts | ✅ | ❌ | ✅ | ❌ |
| Dask Distributed | ✅ | ❌ | ❌ | ❌ |
| Interactive Plotly Reports | ✅ | Partial | ❌ | ✅ |
| Quality Gates for CI/CD | ✅ | ❌ | ✅ | ❌ |

---

## Requirements

### Core Dependencies
- Python 3.8+
- PyQt6 >= 6.6.0
- pandas >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- plotly >= 5.17.0
- scipy >= 1.11.0

### Optional Dependencies
- **SHAP** (shap >= 0.44.0) - For explainability features
- **Dask** (dask[complete] >= 2023.12.0) - For large-scale processing
- **MLflow** (mlflow >= 2.9.0) - For experiment tracking
- **Weights & Biases** (wandb >= 0.16.0) - For experiment tracking
- **WeasyPrint** (weasyprint >= 60.0) - For PDF report generation

---

## Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [CLI Reference](docs/cli_reference.md)
- [Examples](examples/)

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use DataAiPrep in your research, please cite:

```bibtex
@software{dataiprep2024,
  author = {Massaoudi, Mohamed},
  title = {DataAiPrep: Advanced ML Data Quality Assessment Platform},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/massaoudi-lab/dataiprep}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- 📧 Email: mohamed.massaoudi@tamu.edu
- 🐛 Issues: [GitHub Issues](https://github.com/massaoudi-lab/dataiprep/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/massaoudi-lab/dataiprep/discussions)

---

<p align="center">
  Made with ❤️ by the DataAiPrep Team
</p>

