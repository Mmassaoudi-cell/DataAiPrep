#!/usr/bin/env python3
"""
Interactive Report Generation Example

This example demonstrates how to use DataAiPrep to generate interactive
HTML reports with Plotly visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced import (
    InteractiveReportGenerator,
    AdvancedMissingAnalyzer,
    EnsembleOutlierDetector,
    ModelCardGenerator
)


def create_sample_data():
    """Create a sample dataset for report generation."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.normal(650, 80, n_samples).astype(int),
        'account_age_months': np.random.exponential(36, n_samples).astype(int),
        'num_products': np.random.poisson(2, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'is_active': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'churn': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_idx = np.random.choice(n_samples, size=80, replace=False)
    df.loc[missing_idx[:40], 'income'] = np.nan
    df.loc[missing_idx[40:], 'credit_score'] = np.nan
    
    return df


def main():
    """Generate an interactive data quality report."""
    print("=" * 60)
    print("DataAiPrep - Interactive Report Generation Example")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample dataset...")
    df = create_sample_data()
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Run analysis
    print("\n🔍 Running data quality analysis...")
    
    # Missing value analysis
    missing_analyzer = AdvancedMissingAnalyzer()
    missing_results = missing_analyzer.analyze(df)
    
    # Outlier detection
    outlier_detector = EnsembleOutlierDetector(methods=['iqr', 'zscore', 'isolation_forest'])
    outlier_results = outlier_detector.detect(df)
    
    print("   ✅ Analysis complete")
    
    # Create interactive report
    print("\n📊 Generating interactive report...")
    
    report = InteractiveReportGenerator(
        title="Customer Data Quality Report",
        theme="dark"  # or "light"
    )
    
    # Add summary metrics
    report.add_summary_metrics({
        'Total Customers': len(df),
        'Active Customers': df['is_active'].sum(),
        'Churn Rate': f"{df['churn'].mean() * 100:.1f}%",
        'Avg Income': f"${df['income'].mean():,.0f}",
        'Data Completeness': f"{(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%"
    })
    
    # Add quality scores
    completeness_score = 100 - missing_results['summary']['total_missing_percentage']
    outlier_score = max(0, 100 - outlier_results['summary']['consensus_outliers_percentage'] * 10)
    
    report.add_quality_scores({
        'Completeness': completeness_score,
        'Data Quality': outlier_score,
        'Consistency': 92.5,  # Example score
        'Timeliness': 98.0   # Example score
    })
    
    # Add alerts
    alerts = []
    
    if missing_results['summary']['total_missing_percentage'] > 5:
        alerts.append({
            'severity': 'warning',
            'title': 'Missing Data Alert',
            'message': f"{missing_results['summary']['total_missing_percentage']:.1f}% of data is missing. "
                      f"Consider imputation strategies."
        })
    
    if outlier_results['summary']['consensus_outliers_count'] > 20:
        alerts.append({
            'severity': 'warning',
            'title': 'Outliers Detected',
            'message': f"{outlier_results['summary']['consensus_outliers_count']} outliers detected "
                      f"({outlier_results['summary']['consensus_outliers_percentage']:.1f}% of data)"
        })
    
    alerts.append({
        'severity': 'info',
        'title': 'Report Generated',
        'message': f"Analysis completed on {len(df)} records with {len(df.columns)} features."
    })
    
    report.add_alerts(alerts)
    
    # Add feature importance (example)
    feature_importance = {
        'income': 0.35,
        'credit_score': 0.25,
        'account_age_months': 0.18,
        'num_products': 0.12,
        'is_active': 0.07,
        'age': 0.03
    }
    report.add_feature_importance(feature_importance, "Feature Importance for Churn Prediction")
    
    # Add missing data breakdown
    if missing_results['summary'].get('missing_by_column'):
        missing_by_col = {
            col: info['percentage']
            for col, info in missing_results['summary']['missing_by_column'].items()
            if info['percentage'] > 0
        }
        if missing_by_col:
            report.add_feature_importance(missing_by_col, "Missing Data by Column (%)")
    
    # Save HTML report
    output_path = Path("customer_quality_report.html")
    saved_path = report.save(str(output_path), format='html')
    print(f"   ✅ HTML report saved to: {saved_path}")
    
    # Generate Model Card
    print("\n📋 Generating model card...")
    
    model_card = ModelCardGenerator()
    model_card.set_model_details(
        name="Customer Churn Prediction Model",
        version="1.0.0",
        type="Random Forest Classifier",
        description="Predicts customer churn probability based on demographic and behavioral features."
    )
    
    model_card.set_intended_use(
        primary_uses=["Customer retention targeting", "Proactive outreach campaigns"],
        out_of_scope=["Credit decisioning", "Pricing optimization"]
    )
    
    model_card.set_training_data(
        dataset_name="Customer Dataset",
        size=len(df),
        features=list(df.columns)
    )
    
    model_card.set_metrics({
        'accuracy': 0.87,
        'precision': 0.82,
        'recall': 0.79,
        'f1_score': 0.80,
        'auc_roc': 0.91
    })
    
    model_card.set_ethical_considerations([
        "Model should not be used as sole decision-maker for customer actions",
        "Regular fairness audits recommended across demographic groups",
        "Predictions should be reviewed before high-impact decisions"
    ])
    
    model_card_path = Path("model_card.md")
    model_card.save(str(model_card_path))
    print(f"   ✅ Model card saved to: {model_card_path}")
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)
    print(f"\nOpen {output_path} in a browser to view the interactive report.")


if __name__ == "__main__":
    main()

