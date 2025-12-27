"""
MLflow Integration for DataAiPrep

Enables logging data quality metrics to MLflow experiments.

Usage:
    from src.integrations import DataAiPrepMLflow
    
    # Initialize integration
    dap_mlflow = DataAiPrepMLflow(experiment_name="my_experiment")
    
    # Log data quality assessment
    with dap_mlflow.start_run(run_name="data_quality_check"):
        results = dap_mlflow.assess_and_log(df, target_column="target")
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class DataAiPrepMLflow:
    """
    MLflow integration for DataAiPrep data quality assessment.
    
    Features:
    - Log data quality metrics as MLflow metrics
    - Save quality reports as artifacts
    - Track data quality across experiments
    - Register data quality issues as tags
    """
    
    def __init__(
        self,
        experiment_name: str = "dataaiprep_quality",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow integration.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI (optional)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow = None
        self._run = None
        
    def _ensure_mlflow(self):
        """Lazy import MLflow"""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                    
                mlflow.set_experiment(self.experiment_name)
            except ImportError:
                raise ImportError(
                    "MLflow is required for this integration. "
                    "Install with: pip install mlflow"
                )
        return self._mlflow
    
    def start_run(self, run_name: Optional[str] = None, **kwargs):
        """
        Start an MLflow run context.
        
        Args:
            run_name: Name for the run
            **kwargs: Additional arguments for mlflow.start_run()
            
        Returns:
            MLflow run context manager
        """
        mlflow = self._ensure_mlflow()
        return mlflow.start_run(run_name=run_name, **kwargs)
    
    def assess_and_log(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        log_artifacts: bool = True
    ) -> Dict[str, Any]:
        """
        Run data quality assessment and log results to MLflow.
        
        Args:
            data: DataFrame to assess
            target_column: Name of target column
            log_artifacts: Whether to save detailed reports as artifacts
            
        Returns:
            Dictionary of assessment results
        """
        mlflow = self._ensure_mlflow()
        
        # Import analyzers
        from ..analysis.completeness_analyzer import CompletenessAnalyzer
        from ..analysis.distribution_analyzer import DistributionAnalyzer
        from ..analysis.leakage_detector import LeakageDetector
        from ..analysis.feature_quality_analyzer import FeatureQualityAnalyzer
        
        # Run assessments
        completeness = CompletenessAnalyzer().analyze(data)
        distribution = DistributionAnalyzer().analyze(data, target_column)
        leakage = LeakageDetector().analyze(data, target_column)
        feature_quality = FeatureQualityAnalyzer().analyze(data, target_column)
        
        # Log metrics
        mlflow.log_metric("data_rows", len(data))
        mlflow.log_metric("data_columns", len(data.columns))
        mlflow.log_metric("completeness_score", completeness.get('completeness_score', 0))
        mlflow.log_metric("missing_percentage", completeness.get('overall_missing_percentage', 0))
        mlflow.log_metric("leakage_score", leakage.get('leakage_score', 0))
        mlflow.log_metric("num_outlier_columns", len(distribution.get('outliers', {})))
        
        # Log tags for issues
        if completeness.get('overall_missing_percentage', 0) > 5:
            mlflow.set_tag("data_issue.missing_values", "high")
        if leakage.get('leakage_score', 0) > 50:
            mlflow.set_tag("data_issue.leakage_risk", "high")
        if len(leakage.get('perfect_correlations', [])) > 0:
            mlflow.set_tag("data_issue.perfect_correlations", "detected")
        
        # Log parameters
        mlflow.log_param("target_column", target_column or "None")
        mlflow.log_param("assessment_modules", "completeness,distribution,leakage,feature_quality")
        
        results = {
            'completeness': completeness,
            'distribution': distribution,
            'leakage': leakage,
            'feature_quality': feature_quality
        }
        
        # Log artifacts
        if log_artifacts:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save JSON report
                report_path = Path(tmpdir) / "data_quality_report.json"
                with open(report_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                mlflow.log_artifact(str(report_path))
                
                # Save summary
                summary_path = Path(tmpdir) / "quality_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write(f"Data Quality Assessment Summary\n")
                    f.write(f"{'='*40}\n")
                    f.write(f"Rows: {len(data)}\n")
                    f.write(f"Columns: {len(data.columns)}\n")
                    f.write(f"Completeness Score: {completeness.get('completeness_score', 0):.1f}%\n")
                    f.write(f"Missing Percentage: {completeness.get('overall_missing_percentage', 0):.1f}%\n")
                    f.write(f"Leakage Risk Score: {leakage.get('leakage_score', 0):.1f}/100\n")
                mlflow.log_artifact(str(summary_path))
        
        return results
    
    def log_preprocessing_step(
        self,
        step_name: str,
        before_shape: tuple,
        after_shape: tuple,
        changes: Dict[str, Any]
    ):
        """
        Log a preprocessing step to MLflow.
        
        Args:
            step_name: Name of the preprocessing step
            before_shape: Shape before transformation
            after_shape: Shape after transformation
            changes: Dictionary of changes made
        """
        mlflow = self._ensure_mlflow()
        
        mlflow.log_param(f"preprocess.{step_name}.before_rows", before_shape[0])
        mlflow.log_param(f"preprocess.{step_name}.before_cols", before_shape[1])
        mlflow.log_param(f"preprocess.{step_name}.after_rows", after_shape[0])
        mlflow.log_param(f"preprocess.{step_name}.after_cols", after_shape[1])
        
        for key, value in changes.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"preprocess.{step_name}.{key}", value)
            else:
                mlflow.set_tag(f"preprocess.{step_name}.{key}", str(value)[:250])


# Example usage code snippet
MLFLOW_EXAMPLE = '''
# MLflow Integration Example
# ==========================

import mlflow
import pandas as pd
from src.integrations import DataAiPrepMLflow

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize DataAiPrep MLflow integration
dap_mlflow = DataAiPrepMLflow(
    experiment_name="my_ml_project",
    tracking_uri="http://localhost:5000"  # Optional: MLflow server
)

# Run assessment and log to MLflow
with dap_mlflow.start_run(run_name="data_quality_assessment"):
    results = dap_mlflow.assess_and_log(
        data=df,
        target_column="target",
        log_artifacts=True
    )
    
    # Continue with model training in the same run
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Log model performance
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

print("Run logged to MLflow!")
'''

