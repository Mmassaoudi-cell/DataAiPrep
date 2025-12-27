# DataAiPrep User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [GUI Application](#gui-application)
3. [Command Line Interface](#command-line-interface)
4. [Python API](#python-api)
5. [Analysis Modules](#analysis-modules)
6. [Advanced Features](#advanced-features)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
# Basic installation
pip install dataiprep

# Full installation with all features
pip install dataiprep[full]

# From source
git clone https://github.com/massaoudi-lab/dataiprep.git
cd dataiprep
pip install -e .
```

### Quick Start

```python
from src.advanced import DataQualityPipeline
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Create and run pipeline
pipeline = DataQualityPipeline(name="QuickAnalysis")
pipeline.add_step('completeness')
pipeline.add_step('outlier_detection')
pipeline.run(data=df)

# View results
print(pipeline.get_summary())
```

---

## GUI Application

### Launching the GUI

```bash
python main.py
# or
python main.py --gui
```

### Main Window Features

1. **Data Import**: Drag and drop or use File menu to load CSV, Excel, or Parquet files
2. **Analysis Tabs**: View results for different analysis modules
3. **Export**: Save reports in HTML, JSON, or PDF format

### Running Analysis

1. Load your data file
2. Select target column (optional, for supervised analysis)
3. Choose analysis modules
4. Click "Analyze" button
5. View results in respective tabs

---

## Command Line Interface

### Basic Usage

```bash
# Analyze a CSV file
python main.py analyze data.csv --target label --output report.html

# Specify modules
python main.py analyze data.csv --modules completeness,outliers,drift

# With baseline for drift detection
python main.py analyze current.csv --baseline train.csv --target label
```

### Available Commands

| Command | Description |
|---------|-------------|
| `analyze` | Run data quality analysis |
| `select-features` | Advanced feature selection |
| `detect-leakage` | Train-test leakage detection |
| `report` | Generate interactive report |

### Feature Selection

```bash
python main.py select-features data.csv \
    --target label \
    --methods boruta,rfe,lasso,mrmr \
    --n-features 20 \
    --output features.json
```

### Leakage Detection

```bash
python main.py detect-leakage train.csv test.csv \
    --target label \
    --entity customer_id \
    --output leakage_report.json
```

---

## Python API

### DataQualityPipeline

The main entry point for comprehensive analysis:

```python
from src.advanced import DataQualityPipeline

pipeline = DataQualityPipeline(name="MyAnalysis")

# Add analysis steps
pipeline.add_step('completeness', threshold=0.95)
pipeline.add_step('outlier_detection', methods=['iqr', 'zscore', 'isolation_forest'])
pipeline.add_step('drift_detection', baseline=train_data)
pipeline.add_step('fairness', protected_cols=['gender', 'race'])

# Run pipeline
pipeline.run(data=df, target='label', verbose=True)

# Get results
summary = pipeline.get_summary()
recommendations = pipeline.get_recommendations()

# Export
pipeline.export('report.html', format='html')
```

### Individual Analyzers

```python
from src.advanced import (
    AdvancedMissingAnalyzer,
    EnsembleOutlierDetector,
    DriftAnalyzer,
    FairnessAnalyzer
)

# Missing value analysis
analyzer = AdvancedMissingAnalyzer()
results = analyzer.analyze(df)

# Outlier detection
detector = EnsembleOutlierDetector(methods=['iqr', 'zscore', 'isolation_forest'])
results = detector.detect(df)

# Drift detection
drift = DriftAnalyzer()
results = drift.analyze(current_data, baseline_data)

# Fairness analysis
fairness = FairnessAnalyzer(protected_columns=['gender'])
results = fairness.analyze(df, target='label')
```

---

## Analysis Modules

### Completeness Analysis

Analyzes missing values and their patterns:

- Total missing percentage
- Missing by column
- MCAR/MAR/MNAR classification
- Imputation recommendations

### Outlier Detection

Multi-method ensemble approach:

- IQR (Interquartile Range)
- Z-score
- Isolation Forest
- Local Outlier Factor (LOF)
- DBSCAN clustering
- Consensus voting

### Drift Detection

Detects distribution changes:

- Population Stability Index (PSI)
- Kolmogorov-Smirnov test
- Jensen-Shannon divergence
- Chi-square test (categorical)

### Fairness Analysis

Bias and fairness assessment:

- Class imbalance
- Difference in proportions
- Proxy detection
- Intersectional analysis

---

## Advanced Features

### Feature Selection

```python
from src.advanced import AdvancedFeatureSelector

selector = AdvancedFeatureSelector(random_state=42)
results = selector.select_features(
    X, y,
    methods=['boruta', 'rfe', 'lasso', 'mrmr'],
    n_features=20,
    consensus_threshold=0.5
)

# Get consensus features
consensus = results['consensus_features']
```

### SHAP Explainability

```python
from src.advanced import SHAPExplainer

explainer = SHAPExplainer(random_state=42)
results = explainer.explain(X, y)

# Global importance
for feat, importance in results['global_importance'].items():
    print(f"{feat}: {importance:.4f}")
```

### Leakage Detection

```python
from src.advanced import AdvancedLeakageDetector

detector = AdvancedLeakageDetector()
results = detector.detect_all(
    train_data,
    test_data,
    target_column='label',
    entity_column='customer_id'
)

if results['summary']['has_contamination']:
    print("Warning: Train-test contamination detected!")
```

### Scalability with Dask

```python
from src.advanced import LargeDataProcessor

processor = LargeDataProcessor(
    chunk_size=100000,
    use_dask=True,
    n_workers=4
)

df = processor.read_large_csv("large_file.csv", optimize_memory=True)
```

### Quality Gates

```python
from src.advanced import QualityGate

gate = QualityGate(name="ProductionReadiness")
gate.add_check('completeness', 'completeness_score', 'gte', 95, severity='error')
gate.add_check('outliers', 'outlier_percentage', 'lte', 2, severity='warning')

result = gate.evaluate(metrics)
if not result['passed']:
    print("Quality gate failed!")
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATAIPREP_LOG_LEVEL` | Logging level | `INFO` |
| `DATAIPREP_THEME` | GUI theme | `dark` |
| `DATAIPREP_WORKERS` | Dask workers | `4` |

### Configuration File

Create `dataiprep.yaml`:

```yaml
analysis:
  completeness_threshold: 0.95
  outlier_methods:
    - iqr
    - zscore
    - isolation_forest
  
reporting:
  theme: dark
  format: html
  
alerting:
  slack_webhook: "https://hooks.slack.com/..."
  email_recipients:
    - team@example.com
```

---

## Troubleshooting

### Common Issues

**PyQt6 not found**
```bash
pip install PyQt6 PyQt6-SVG
```

**SHAP not working**
```bash
pip install shap
```

**Memory errors with large files**
```python
from src.advanced import LargeDataProcessor

processor = LargeDataProcessor(use_dask=True)
df = processor.read_large_csv("large_file.csv")
```

**Import errors**
```bash
# Ensure you're in the correct directory
cd dataiprep
pip install -e .
```

### Getting Help

- [GitHub Issues](https://github.com/massaoudi-lab/dataiprep/issues)
- [Discussions](https://github.com/massaoudi-lab/dataiprep/discussions)
- Email: mohamed.massaoudi@tamu.edu

