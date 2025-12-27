#!/usr/bin/env python3
"""
DataAiPrep - Advanced ML Data Quality Assessment Platform

A comprehensive, enterprise-grade data preprocessing and quality assessment tool
that goes beyond PyCaret's capabilities with expert-level features.

Features:
- Advanced Missing Value Analysis (MCAR/MAR/MNAR classification)
- Ensemble Outlier Detection (IQR, Z-score, Isolation Forest, LOF, DBSCAN)
- Data Drift Detection (PSI, KS test, JS divergence)
- Fairness & Bias Analysis
- Automated Feature Engineering
- Model Explainability (SHAP integration)
- Time Series Analysis
- Advanced Feature Selection (Boruta, RFE, LASSO, mRMR)
- Scalability (Dask, chunked processing)
- Alerting (Slack, Email, Webhooks)
- Interactive Reports (Plotly, PDF export)
- Advanced Leakage Detection (train-test contamination)
- Full Pipeline API

Usage:
    # GUI Application
    python main.py
    python main.py --gui
    
    # CLI Quality Analysis
    python main.py analyze data.csv --target label --output report.html
    
    # Specific Modules
    python main.py analyze data.csv --modules completeness,outliers,drift
    
    # Compare with baseline (drift detection)
    python main.py analyze current.csv --baseline train.csv --target label
    
    # Fairness analysis
    python main.py analyze data.csv --target label --protected gender,race
    
    # Feature selection
    python main.py select-features data.csv --target label --methods boruta,rfe,lasso
    
    # Leakage detection
    python main.py detect-leakage train.csv test.csv --target label
    
    # Generate interactive report
    python main.py report data.csv --output report.html --format html
    
    # Web Demo
    python main.py --web --port 8000
    
    # Generate Benchmark Datasets
    python main.py --benchmark --output benchmark_datasets/
    
    # Show Examples
    python main.py --examples
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional
import json

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_gui():
    """Launch the PyQt6 GUI application"""
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QIcon
    except ImportError as e:
        print("❌ Error: PyQt6 is required for the GUI.")
        print("   Install with: pip install PyQt6 PyQt6-SVG")
        print(f"   Details: {e}")
        sys.exit(1)
    
    try:
        from src.gui.main_window import MainWindow
    except ImportError as e:
        print(f"❌ Error importing GUI module: {e}")
        sys.exit(1)

    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("DataAiPrep")
    app.setApplicationVersion("2.0.0")
    app.setApplicationDisplayName("DataAiPrep - Advanced ML Data Quality Assessment")
    app.setOrganizationName("DataAiPrep")
    
    # Set application icon
    _set_app_icon(app)
    
    # Create and show main window
    try:
        window = MainWindow()
        window.show()
        print("🚀 DataAiPrep GUI launched successfully!")
        return app.exec()
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        _show_error_dialog(app, f"Failed to launch DataAiPrep GUI:\n\n{e}")
        return 1


def _set_app_icon(app):
    """Set application icon from logo file"""
    try:
        from PyQt6.QtGui import QPixmap, QPainter, QIcon
        from PyQt6.QtCore import Qt
        
        # Try to load SVG support
        try:
            from PyQt6.QtSvg import QSvgRenderer
            HAS_SVG = True
        except ImportError:
            HAS_SVG = False
            
        logo_path = Path(__file__).parent / "data_ai_prep_logo.svg"
        
        if HAS_SVG and logo_path.exists():
            renderer = QSvgRenderer(str(logo_path))
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            app.setWindowIcon(QIcon(pixmap))
            print("🎨 Application icon loaded!")
    except Exception as e:
        # Graceful fallback to default icon
        pass


def _show_error_dialog(app, message: str):
    """Show error dialog if possible"""
    try:
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("DataAiPrep Error")
        msg.setText(message)
        msg.exec()
    except Exception:
        pass


def run_cli_analysis(
    data_file: str,
    target: str = None,
    output: str = None,
    format: str = 'html',
    modules: List[str] = None,
    baseline: str = None,
    protected: List[str] = None,
    verbose: bool = True
):
    """Run comprehensive CLI data quality analysis"""
    import pandas as pd
    
    print("\n" + "="*60)
    print("🔍 DATAAIPREP - Advanced Data Quality Analysis")
    print("="*60 + "\n")
    
    # Load data
    try:
        print(f"📂 Loading data from: {data_file}")
        data = pd.read_csv(data_file)
        print(f"   Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Load baseline if provided
    baseline_data = None
    if baseline:
        try:
            print(f"📂 Loading baseline from: {baseline}")
            baseline_data = pd.read_csv(baseline)
            print(f"   Shape: {baseline_data.shape[0]} rows × {baseline_data.shape[1]} columns")
        except Exception as e:
            print(f"⚠️  Warning: Could not load baseline: {e}")
    
    # Default modules
    if modules is None:
        modules = ['completeness', 'outlier_detection']
        if baseline_data is not None:
            modules.append('drift_detection')
        if protected and target:
            modules.append('fairness')
        if target:
            modules.append('explainability')
    
    print(f"\n📋 Modules to run: {', '.join(modules)}")
    
    # Import and run pipeline
    try:
        from src.advanced import DataQualityPipeline
        
        pipeline = DataQualityPipeline(name="CLI Analysis")
        
        # Add steps based on modules
        for module in modules:
            if module == 'completeness':
                pipeline.add_step('completeness')
            elif module in ['outliers', 'outlier_detection']:
                pipeline.add_step('outlier_detection', 
                                 methods=['iqr', 'zscore', 'isolation_forest', 'lof'])
            elif module in ['drift', 'drift_detection']:
                if baseline_data is not None:
                    pipeline.add_step('drift_detection', baseline=baseline_data)
                else:
                    print(f"⚠️  Skipping drift detection: no baseline provided")
            elif module == 'fairness':
                if protected and target:
                    pipeline.add_step('fairness', protected_cols=protected)
                else:
                    print(f"⚠️  Skipping fairness: requires --protected and --target")
            elif module in ['explain', 'explainability']:
                if target:
                    pipeline.add_step('explainability')
                else:
                    print(f"⚠️  Skipping explainability: requires --target")
            elif module in ['features', 'feature_engineering']:
                pipeline.add_step('feature_engineering')
            else:
                print(f"⚠️  Unknown module: {module}")
        
        # Run pipeline
        print("\n")
        pipeline.run(data, target=target, verbose=verbose)
        
        # Export results
        if output:
            print(f"\n📄 Exporting report to: {output}")
            pipeline.export(output, format=format)
            print(f"✅ Report saved!")
        
        # Print summary
        summary = pipeline.get_summary()
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        print(f"Status: {'✅ SUCCESS' if summary['success'] else '❌ FAILED'}")
        print(f"Steps Executed: {len(summary['steps_executed'])}")
        print(f"Execution Time: {summary['execution_time']:.2f}s")
        print(f"Total Recommendations: {summary['total_recommendations']}")
        print(f"High Severity Issues: {summary['high_severity_issues']}")
        
        # Print recommendations
        recommendations = pipeline.get_recommendations()
        if recommendations:
            print("\n" + "-"*60)
            print("🔔 TOP RECOMMENDATIONS")
            print("-"*60)
            for i, rec in enumerate(recommendations[:10], 1):
                severity = rec.get('severity', 'medium').upper()
                category = rec.get('category', 'General')
                issue = rec.get('issue', rec.get('message', str(rec)))
                print(f"\n{i}. [{severity}] {category}")
                print(f"   {issue[:100]}{'...' if len(issue) > 100 else ''}")
        
        return 0 if summary['success'] else 1
        
    except ImportError as e:
        print(f"❌ Error importing advanced modules: {e}")
        print("   Make sure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_feature_selection(
    data_file: str,
    target: str,
    methods: List[str] = None,
    n_features: int = None,
    output: str = None,
    verbose: bool = True
):
    """Run advanced feature selection"""
    import pandas as pd
    
    print("\n" + "="*60)
    print("🎯 DATAAIPREP - Advanced Feature Selection")
    print("="*60 + "\n")
    
    try:
        print(f"📂 Loading data from: {data_file}")
        data = pd.read_csv(data_file)
        print(f"   Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    if target not in data.columns:
        print(f"❌ Target column '{target}' not found in data")
        return 1
    
    X = data.drop(columns=[target])
    y = data[target]
    
    try:
        from src.advanced import AdvancedFeatureSelector
        
        selector = AdvancedFeatureSelector()
        
        print(f"\n📋 Methods: {', '.join(methods) if methods else 'all'}")
        print(f"🎯 Target: {target}")
        
        results = selector.select_features(
            X, y, 
            methods=methods,
            n_features=n_features
        )
        
        print("\n" + selector.generate_report())
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✅ Results saved to: {output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during feature selection: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_leakage_detection(
    train_file: str,
    test_file: str,
    target: str = None,
    entity_column: str = None,
    output: str = None,
    verbose: bool = True
):
    """Run advanced leakage detection"""
    import pandas as pd
    
    print("\n" + "="*60)
    print("🔍 DATAAIPREP - Advanced Leakage Detection")
    print("="*60 + "\n")
    
    try:
        print(f"📂 Loading train data from: {train_file}")
        train_data = pd.read_csv(train_file)
        print(f"   Shape: {train_data.shape[0]} rows × {train_data.shape[1]} columns")
        
        print(f"📂 Loading test data from: {test_file}")
        test_data = pd.read_csv(test_file)
        print(f"   Shape: {test_data.shape[0]} rows × {test_data.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    try:
        from src.advanced import AdvancedLeakageDetector
        
        detector = AdvancedLeakageDetector()
        
        results = detector.detect_all(
            train_data,
            test_data,
            target_column=target,
            entity_column=entity_column
        )
        
        print("\n" + detector.generate_report())
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✅ Results saved to: {output}")
        
        return 0 if results['summary']['total_issues'] == 0 else 1
        
    except Exception as e:
        print(f"❌ Error during leakage detection: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_interactive_report(
    data_file: str,
    output: str,
    format: str = 'html',
    target: str = None,
    theme: str = 'dark',
    verbose: bool = True
):
    """Generate an interactive report"""
    import pandas as pd
    
    print("\n" + "="*60)
    print("📊 DATAAIPREP - Interactive Report Generator")
    print("="*60 + "\n")
    
    try:
        print(f"📂 Loading data from: {data_file}")
        data = pd.read_csv(data_file)
        print(f"   Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    try:
        from src.advanced import (
            InteractiveReportGenerator, 
            DataQualityPipeline,
            AdvancedMissingAnalyzer,
            EnsembleOutlierDetector
        )
        
        # Run analysis
        print("\n▶️ Running data quality analysis...")
        
        # Completeness
        missing_analyzer = AdvancedMissingAnalyzer()
        missing_results = missing_analyzer.analyze(data)
        
        # Outliers
        outlier_detector = EnsembleOutlierDetector(methods=['iqr', 'zscore', 'isolation_forest'])
        outlier_results = outlier_detector.detect(data)
        
        # Generate report
        print("\n▶️ Generating interactive report...")
        
        report = InteractiveReportGenerator(
            title="DataAiPrep Quality Report",
            theme=theme
        )
        
        # Add summary metrics
        report.add_summary_metrics({
            'Total Rows': len(data),
            'Total Columns': len(data.columns),
            'Missing %': f"{missing_results['summary']['total_missing_percentage']:.1f}%",
            'Outliers': outlier_results['summary']['consensus_outliers_count']
        })
        
        # Add quality scores
        report.add_quality_scores({
            'Completeness': 100 - missing_results['summary']['total_missing_percentage'],
            'Data Quality': max(0, 100 - outlier_results['summary']['consensus_outliers_percentage'] * 5)
        })
        
        # Add alerts
        alerts = []
        if missing_results['summary']['total_missing_percentage'] > 10:
            alerts.append({
                'severity': 'warning',
                'title': 'High Missing Rate',
                'message': f"{missing_results['summary']['total_missing_percentage']:.1f}% of data is missing"
            })
        
        if outlier_results['summary']['consensus_outliers_percentage'] > 5:
            alerts.append({
                'severity': 'warning',
                'title': 'Outliers Detected',
                'message': f"{outlier_results['summary']['consensus_outliers_count']} outliers found ({outlier_results['summary']['consensus_outliers_percentage']:.1f}%)"
            })
        
        if alerts:
            report.add_alerts(alerts)
        
        # Add missing data by column
        if missing_results['summary'].get('missing_by_column'):
            missing_by_col = {
                col: info['percentage'] 
                for col, info in missing_results['summary']['missing_by_column'].items()
                if info['percentage'] > 0
            }
            if missing_by_col:
                report.add_feature_importance(missing_by_col, "Missing Data by Column")
        
        # Save report
        saved_path = report.save(output, format=format)
        print(f"\n✅ Report saved to: {saved_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_web_demo(host: str = "127.0.0.1", port: int = 8000):
    """Launch the web demo application"""
    try:
        from src.web.demo import main as web_main
        
        # Modify sys.argv for the web demo
        original_argv = sys.argv
        sys.argv = ['demo', '--host', host, '--port', str(port)]
        
        web_main()
        
        sys.argv = original_argv
        
    except ImportError as e:
        print("❌ Error: FastAPI is required for the web demo.")
        print("   Install with: pip install fastapi uvicorn python-multipart")
        print(f"   Details: {e}")
        sys.exit(1)


def generate_benchmarks(output_dir: str = None):
    """Generate benchmark datasets"""
    try:
        from src.benchmark import BenchmarkDatasetGenerator
        
        output_path = Path(output_dir) if output_dir else Path("benchmark_datasets")
        
        print("🔧 Generating benchmark datasets...")
        generator = BenchmarkDatasetGenerator(random_state=42)
        datasets = generator.generate_all_benchmarks(save_path=output_path)
        
        print(f"\n✅ Generated {len(datasets)} benchmark datasets!")
        print(f"📁 Saved to: {output_path.absolute()}")
        
        print("\n" + "="*60)
        print("BENCHMARK DATASET SUMMARY")
        print("="*60)
        print(generator.get_dataset_summary().to_string(index=False))
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating benchmarks: {e}")
        return 1


def show_examples():
    """Show integration examples"""
    print("""
================================================================================
DataAiPrep Integration Examples - Advanced Features
================================================================================

📊 1. COMPREHENSIVE PIPELINE API
---------------------------------
from src.advanced import DataQualityPipeline
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")
train_df = pd.read_csv("training_data.csv")  # For drift detection

# Create and configure pipeline
pipeline = DataQualityPipeline(name="MyAnalysis")
pipeline.add_step('completeness', threshold=0.95)
pipeline.add_step('outlier_detection', methods=['iqr', 'isolation_forest', 'lof'])
pipeline.add_step('drift_detection', baseline=train_df)
pipeline.add_step('fairness', protected_cols=['gender', 'race'])
pipeline.add_step('explainability')

# Execute pipeline
pipeline.run(data=df, target='label')

# Export results
pipeline.export('quality_report.html', format='html')
pipeline.export('quality_report.json', format='json')

# Get recommendations
for rec in pipeline.get_recommendations(severity='high'):
    print(f"[{rec['severity']}] {rec['issue']}")

================================================================================

🎯 2. ADVANCED FEATURE SELECTION (NEW!)
----------------------------------------
from src.advanced import AdvancedFeatureSelector
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=['target'])
y = df['target']

# Run multiple selection methods
selector = AdvancedFeatureSelector(random_state=42)
results = selector.select_features(
    X, y,
    methods=['boruta', 'rfe', 'lasso', 'mrmr', 'variance', 'correlation'],
    n_features=20,
    consensus_threshold=0.5
)

# Get consensus features (selected by multiple methods)
consensus_features = results['consensus_features']
print(f"Features selected by consensus: {consensus_features}")

# Get optimal features (selected by at least 2 methods)
optimal_features = selector.get_optimal_features(min_methods=2)

# Feature rankings
for feat, score in list(results['feature_rankings'].items())[:10]:
    print(f"{feat}: {score:.4f}")

================================================================================

🧠 3. SHAP EXPLAINABILITY (NEW!)
---------------------------------
from src.advanced import SHAPExplainer
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=['target'])
y = df['target']

# Initialize explainer
explainer = SHAPExplainer(random_state=42)

# Generate SHAP explanations
results = explainer.explain(X, y)

# Global feature importance
for feat, importance in list(results['global_importance'].items())[:10]:
    print(f"{feat}: {importance:.4f}")

# Explain a single prediction
instance = X.iloc[[0]]
explanation = explainer.explain_instance(instance)
print(explanation)

# Get summary data for visualization
summary = explainer.get_summary_data()

================================================================================

🔍 4. ADVANCED LEAKAGE DETECTION (NEW!)
----------------------------------------
from src.advanced import AdvancedLeakageDetector
import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

detector = AdvancedLeakageDetector(
    similarity_threshold=0.95,
    correlation_threshold=0.95
)

# Detect all types of leakage
results = detector.detect_all(
    train_data=train_df,
    test_data=test_df,
    target_column='label',
    entity_column='customer_id'  # For group leakage detection
)

# Check results
if results['summary']['has_contamination']:
    print("⚠️ Train-test contamination detected!")
    print(f"   Duplicates: {results['train_test_contamination']['exact_duplicates']}")

if results['summary']['has_group_leakage']:
    print("⚠️ Group leakage detected!")
    print(f"   Overlapping entities: {results['group_leakage']['overlap_count']}")

# Simulate removal of suspected leaky features
impact = detector.simulate_leakage_removal(
    train_df, 'label',
    suspected_features=['suspicious_feature']
)
print(f"Impact of removing feature: {impact}")

================================================================================

📈 5. SCALABILITY - LARGE DATA PROCESSING (NEW!)
-------------------------------------------------
from src.advanced import LargeDataProcessor, IntelligentSampler
import pandas as pd

# Initialize processor
processor = LargeDataProcessor(
    chunk_size=100000,
    use_dask=True,
    n_workers=4
)

# Read large CSV efficiently
df = processor.read_large_csv(
    "large_file.csv",
    optimize_memory=True
)

# Process in chunks
def process_chunk(chunk):
    return chunk.describe()

results = processor.process_in_chunks(
    "large_file.csv",
    process_func=process_chunk
)

# Optimize memory usage
df_optimized = processor.optimize_memory(df, verbose=True)

# Get memory statistics
memory_stats = processor.get_memory_usage(df)
print(f"Memory usage: {memory_stats['total_mb']:.2f} MB")

# Intelligent sampling
sampler = IntelligentSampler(random_state=42)
sample = sampler.stratified_sample(df, 'target', frac=0.1)

================================================================================

🔔 6. ALERTING & QUALITY GATES (NEW!)
--------------------------------------
from src.advanced import AlertManager, SlackChannel, QualityGate

# Set up alert manager
alert_manager = AlertManager()

# Add Slack channel
slack = SlackChannel(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    channel="#data-quality"
)
alert_manager.add_channel('slack', slack)

# Configure thresholds
alert_manager.add_threshold('missing_percentage', 'gt', 10, severity='warning')
alert_manager.add_threshold('outlier_percentage', 'gt', 5, severity='error')

# Check metrics and send alerts
metrics = {'missing_percentage': 15, 'outlier_percentage': 3}
triggered = alert_manager.check_thresholds(metrics, send_alerts=True)

# Quality Gate
gate = QualityGate(name="ProductionReadiness")
gate.add_check('completeness', 'completeness_score', 'gte', 95, severity='error')
gate.add_check('outliers', 'outlier_percentage', 'lte', 2, severity='warning')

result = gate.evaluate({'completeness_score': 98, 'outlier_percentage': 1.5})
print(f"Quality Gate: {'PASSED' if result['passed'] else 'FAILED'}")

================================================================================

📊 7. INTERACTIVE REPORTS (NEW!)
---------------------------------
from src.advanced import InteractiveReportGenerator, ModelCardGenerator
import pandas as pd

# Generate interactive HTML report
report = InteractiveReportGenerator(
    title="Data Quality Report",
    theme="dark"  # or "light"
)

# Add summary metrics
report.add_summary_metrics({
    'Total Rows': 10000,
    'Total Columns': 50,
    'Missing %': '5.2%'
})

# Add quality score gauges
report.add_quality_scores({
    'Completeness': 94.8,
    'Distribution Quality': 87.5,
    'Leakage Risk': 12.0
})

# Add alerts
report.add_alerts([
    {'severity': 'warning', 'title': 'High Missing', 'message': '5 columns have >10% missing'},
    {'severity': 'info', 'title': 'Outliers', 'message': '150 outliers detected'}
])

# Add feature importance chart
report.add_feature_importance({'feature_a': 0.35, 'feature_b': 0.25, 'feature_c': 0.2})

# Save report
report.save('quality_report.html', format='html')
report.save('quality_report.pdf', format='pdf')  # Requires weasyprint

# Generate Model Card
model_card = ModelCardGenerator()
model_card.set_model_details(
    name="Fraud Detection Model",
    version="1.0",
    type="RandomForest Classifier",
    description="Detects fraudulent transactions"
)
model_card.set_metrics({'accuracy': 0.95, 'precision': 0.92, 'recall': 0.88})
model_card.save('model_card.md')

================================================================================

🚀 8. CLI USAGE
----------------
# Basic analysis
python main.py analyze data.csv --target label --output report.html

# With specific modules
python main.py analyze data.csv --modules completeness,outliers,drift,fairness

# Drift detection with baseline
python main.py analyze production.csv --baseline training.csv --target label

# Feature selection (NEW!)
python main.py select-features data.csv --target label --methods boruta,rfe,lasso

# Leakage detection (NEW!)
python main.py detect-leakage train.csv test.csv --target label --entity customer_id

# Generate interactive report (NEW!)
python main.py report data.csv --output report.html --theme dark

# Multiple output formats
python main.py analyze data.csv --target label --output report --format json

================================================================================
""")
    return 0


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='DataAiPrep - Advanced ML Data Quality Assessment Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Launch GUI
  python main.py analyze data.csv --target label   # CLI analysis
  python main.py analyze data.csv --modules completeness,outliers
  python main.py analyze data.csv --baseline train.csv --target label
  python main.py select-features data.csv --target label --methods boruta,rfe
  python main.py detect-leakage train.csv test.csv --target label
  python main.py report data.csv --output report.html
  python main.py --web                              # Launch web demo
  python main.py --benchmark                        # Generate benchmarks
  python main.py --examples                         # Show examples
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command')
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Run data quality analysis')
    analyze_parser.add_argument('data_file', help='Path to data file (CSV)')
    analyze_parser.add_argument('--target', '-t', help='Target column name')
    analyze_parser.add_argument('--output', '-o', help='Output file path')
    analyze_parser.add_argument('--format', '-f', default='html',
                               choices=['html', 'json', 'text'],
                               help='Output format (default: html)')
    analyze_parser.add_argument('--modules', '-m',
                               help='Comma-separated list of modules to run '
                                    '(completeness,outliers,drift,fairness,explainability,features)')
    analyze_parser.add_argument('--baseline', '-b',
                               help='Baseline data file for drift detection')
    analyze_parser.add_argument('--protected', '-p',
                               help='Comma-separated list of protected columns for fairness analysis')
    analyze_parser.add_argument('--quiet', '-q', action='store_true',
                               help='Suppress verbose output')
    
    # Feature selection subcommand (NEW!)
    select_parser = subparsers.add_parser('select-features', help='Run advanced feature selection')
    select_parser.add_argument('data_file', help='Path to data file (CSV)')
    select_parser.add_argument('--target', '-t', required=True, help='Target column name')
    select_parser.add_argument('--methods', '-m',
                              help='Comma-separated list of methods (boruta,rfe,lasso,elastic_net,mrmr,variance,correlation)')
    select_parser.add_argument('--n-features', '-n', type=int, help='Target number of features')
    select_parser.add_argument('--output', '-o', help='Output JSON file path')
    select_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    
    # Leakage detection subcommand (NEW!)
    leakage_parser = subparsers.add_parser('detect-leakage', help='Run advanced leakage detection')
    leakage_parser.add_argument('train_file', help='Path to training data file (CSV)')
    leakage_parser.add_argument('test_file', help='Path to test data file (CSV)')
    leakage_parser.add_argument('--target', '-t', help='Target column name')
    leakage_parser.add_argument('--entity', '-e', help='Entity column for group leakage detection')
    leakage_parser.add_argument('--output', '-o', help='Output JSON file path')
    leakage_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    
    # Report generation subcommand (NEW!)
    report_parser = subparsers.add_parser('report', help='Generate interactive report')
    report_parser.add_argument('data_file', help='Path to data file (CSV)')
    report_parser.add_argument('--output', '-o', required=True, help='Output file path')
    report_parser.add_argument('--format', '-f', default='html',
                              choices=['html', 'json', 'pdf'],
                              help='Output format (default: html)')
    report_parser.add_argument('--target', '-t', help='Target column name')
    report_parser.add_argument('--theme', default='dark',
                              choices=['dark', 'light'],
                              help='Report theme (default: dark)')
    report_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    
    # Global options
    parser.add_argument('--gui', action='store_true', 
                       help='Launch GUI application (default)')
    parser.add_argument('--web', action='store_true',
                       help='Launch web demo')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Web server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Web server port (default: 8000)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Generate benchmark datasets')
    parser.add_argument('--benchmark-output', default='benchmark_datasets',
                       help='Output directory for benchmarks')
    parser.add_argument('--examples', action='store_true',
                       help='Show integration examples')
    parser.add_argument('--version', action='version', 
                       version='DataAiPrep 2.0.0')
    
    args = parser.parse_args()
    
    # Determine which mode to run
    if args.command == 'analyze':
        modules = args.modules.split(',') if args.modules else None
        protected = args.protected.split(',') if args.protected else None
        return run_cli_analysis(
            data_file=args.data_file,
            target=args.target,
            output=args.output,
            format=args.format,
            modules=modules,
            baseline=args.baseline,
            protected=protected,
            verbose=not args.quiet
        )
    
    if args.command == 'select-features':
        methods = args.methods.split(',') if args.methods else None
        return run_feature_selection(
            data_file=args.data_file,
            target=args.target,
            methods=methods,
            n_features=args.n_features,
            output=args.output,
            verbose=not args.quiet
        )
    
    if args.command == 'detect-leakage':
        return run_leakage_detection(
            train_file=args.train_file,
            test_file=args.test_file,
            target=args.target,
            entity_column=args.entity,
            output=args.output,
            verbose=not args.quiet
        )
    
    if args.command == 'report':
        return run_interactive_report(
            data_file=args.data_file,
            output=args.output,
            format=args.format,
            target=args.target,
            theme=args.theme,
            verbose=not args.quiet
        )
    
    if args.examples:
        return show_examples()
    
    if args.benchmark:
        return generate_benchmarks(args.benchmark_output)
    
    if args.web:
        return run_web_demo(args.host, args.port)
    
    # Default: run GUI
    return run_gui()


# Lazy imports for programmatic use (available when imported as module)
def _get_pipeline():
    from src.advanced import DataQualityPipeline
    return DataQualityPipeline

def _get_missing_analyzer():
    from src.advanced import AdvancedMissingAnalyzer
    return AdvancedMissingAnalyzer

def _get_outlier_detector():
    from src.advanced import EnsembleOutlierDetector
    return EnsembleOutlierDetector

def _get_drift_analyzer():
    from src.advanced import DriftAnalyzer
    return DriftAnalyzer

def _get_fairness_analyzer():
    from src.advanced import FairnessAnalyzer
    return FairnessAnalyzer

def _get_feature_engineer():
    from src.advanced import AdvancedFeatureEngineer
    return AdvancedFeatureEngineer

def _get_explainability():
    from src.advanced import ExplainabilityEngine
    return ExplainabilityEngine

def _get_feature_selector():
    from src.advanced import AdvancedFeatureSelector
    return AdvancedFeatureSelector

def _get_shap_explainer():
    from src.advanced import SHAPExplainer
    return SHAPExplainer

def _get_leakage_detector():
    from src.advanced import AdvancedLeakageDetector
    return AdvancedLeakageDetector

def _get_report_generator():
    from src.advanced import InteractiveReportGenerator
    return InteractiveReportGenerator

def _get_large_data_processor():
    from src.advanced import LargeDataProcessor
    return LargeDataProcessor

def _get_alert_manager():
    from src.advanced import AlertManager
    return AlertManager

def _get_quality_gate():
    from src.advanced import QualityGate
    return QualityGate

__all__ = [
    'run_gui',
    'run_cli_analysis',
    'run_feature_selection',
    'run_leakage_detection',
    'run_interactive_report',
    'run_web_demo',
    '_get_pipeline',
    '_get_missing_analyzer',
    '_get_outlier_detector',
    '_get_drift_analyzer',
    '_get_fairness_analyzer',
    '_get_feature_engineer',
    '_get_explainability',
    '_get_feature_selector',
    '_get_shap_explainer',
    '_get_leakage_detector',
    '_get_report_generator',
    '_get_large_data_processor',
    '_get_alert_manager',
    '_get_quality_gate'
]


if __name__ == "__main__":
    sys.exit(main() or 0)
