"""
Weights & Biases Integration for DataAiPrep

Enables logging data quality metrics to W&B experiments.

Usage:
    from src.integrations import DataAiPrepWandB
    
    # Initialize integration
    dap_wandb = DataAiPrepWandB(project="my_project")
    
    # Log data quality assessment
    dap_wandb.init(name="data_quality_check")
    results = dap_wandb.assess_and_log(df, target_column="target")
    dap_wandb.finish()
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class DataAiPrepWandB:
    """
    Weights & Biases integration for DataAiPrep data quality assessment.
    
    Features:
    - Log data quality metrics to W&B
    - Create quality visualizations
    - Track data quality across runs
    - Generate data quality tables
    """
    
    def __init__(
        self,
        project: str = "dataaiprep-quality",
        entity: Optional[str] = None
    ):
        """
        Initialize W&B integration.
        
        Args:
            project: W&B project name
            entity: W&B entity (team or username)
        """
        self.project = project
        self.entity = entity
        self._wandb = None
        self._run = None
        
    def _ensure_wandb(self):
        """Lazy import wandb"""
        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "Weights & Biases is required for this integration. "
                    "Install with: pip install wandb"
                )
        return self._wandb
    
    def init(
        self,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize a W&B run.
        
        Args:
            name: Run name
            config: Configuration dictionary
            **kwargs: Additional arguments for wandb.init()
        """
        wandb = self._ensure_wandb()
        
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config or {},
            **kwargs
        )
        return self._run
    
    def finish(self):
        """Finish the current W&B run"""
        if self._run:
            self._run.finish()
            self._run = None
    
    def assess_and_log(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        create_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Run data quality assessment and log results to W&B.
        
        Args:
            data: DataFrame to assess
            target_column: Name of target column
            create_visualizations: Whether to create W&B visualizations
            
        Returns:
            Dictionary of assessment results
        """
        wandb = self._ensure_wandb()
        
        if self._run is None:
            self.init()
        
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
        
        # Log summary metrics
        wandb.log({
            "data/rows": len(data),
            "data/columns": len(data.columns),
            "quality/completeness_score": completeness.get('completeness_score', 0),
            "quality/missing_percentage": completeness.get('overall_missing_percentage', 0),
            "quality/leakage_score": leakage.get('leakage_score', 0),
            "quality/num_outlier_columns": len(distribution.get('outliers', {})),
            "quality/perfect_correlations": len(leakage.get('perfect_correlations', [])),
            "quality/duplicate_features": len(leakage.get('duplicate_features', []))
        })
        
        # Log configuration
        wandb.config.update({
            "target_column": target_column or "None",
            "data_shape": list(data.shape),
            "column_dtypes": {str(k): str(v) for k, v in data.dtypes.to_dict().items()}
        })
        
        results = {
            'completeness': completeness,
            'distribution': distribution,
            'leakage': leakage,
            'feature_quality': feature_quality
        }
        
        # Create visualizations
        if create_visualizations:
            self._log_visualizations(data, completeness, distribution, leakage)
        
        # Log data quality table
        self._log_quality_table(completeness, distribution, leakage)
        
        return results
    
    def _log_visualizations(
        self,
        data: pd.DataFrame,
        completeness: Dict,
        distribution: Dict,
        leakage: Dict
    ):
        """Create and log W&B visualizations"""
        wandb = self._ensure_wandb()
        
        # Missing values by column
        missing_by_col = completeness.get('missing_by_column', {})
        if missing_by_col:
            # Get top 20 columns with missing values
            sorted_missing = sorted(
                missing_by_col.items(), 
                key=lambda x: x[1].get('count', 0) if isinstance(x[1], dict) else x[1],
                reverse=True
            )[:20]
            
            if sorted_missing:
                columns = [x[0] for x in sorted_missing]
                values = [
                    x[1].get('percentage', 0) if isinstance(x[1], dict) else 0 
                    for x in sorted_missing
                ]
                
                wandb.log({
                    "charts/missing_values": wandb.plot.bar(
                        wandb.Table(
                            data=[[c, v] for c, v in zip(columns, values)],
                            columns=["Column", "Missing %"]
                        ),
                        "Column",
                        "Missing %",
                        title="Missing Values by Column"
                    )
                })
        
        # Quality score gauge
        completeness_score = completeness.get('completeness_score', 0)
        leakage_score = leakage.get('leakage_score', 0)
        
        quality_data = [
            ["Completeness", completeness_score],
            ["Leakage Risk", 100 - leakage_score],  # Invert so higher is better
        ]
        
        wandb.log({
            "charts/quality_scores": wandb.Table(
                data=quality_data,
                columns=["Metric", "Score"]
            )
        })
    
    def _log_quality_table(
        self,
        completeness: Dict,
        distribution: Dict,
        leakage: Dict
    ):
        """Log summary quality table to W&B"""
        wandb = self._ensure_wandb()
        
        issues = []
        
        # Missing value issues
        if completeness.get('overall_missing_percentage', 0) > 5:
            issues.append(["Missing Values", "High", 
                         f"{completeness.get('overall_missing_percentage', 0):.1f}% missing"])
        
        # Leakage issues
        for corr in leakage.get('perfect_correlations', [])[:5]:
            issues.append(["Perfect Correlation", "Critical",
                         f"{corr[0]} ↔ {corr[1]}"])
        
        # Duplicate features
        for dup in leakage.get('duplicate_features', [])[:5]:
            issues.append(["Duplicate Feature", "Warning",
                         f"{dup[0]} = {dup[1]}"])
        
        if issues:
            wandb.log({
                "quality/issues_table": wandb.Table(
                    data=issues,
                    columns=["Issue Type", "Severity", "Details"]
                )
            })


# Example usage code snippet
WANDB_EXAMPLE = '''
# Weights & Biases Integration Example
# ====================================

import pandas as pd
from src.integrations import DataAiPrepWandB

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize DataAiPrep W&B integration
dap_wandb = DataAiPrepWandB(
    project="my-ml-project",
    entity="my-team"  # Optional
)

# Start a run and assess data quality
dap_wandb.init(
    name="data_quality_assessment",
    config={"dataset": "training_data_v1"}
)

results = dap_wandb.assess_and_log(
    data=df,
    target_column="target",
    create_visualizations=True
)

# Continue with model training in the same run
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log model performance
accuracy = model.score(X_test, y_test)
wandb.log({"accuracy": accuracy})

# Finish the run
dap_wandb.finish()

print("Run logged to Weights & Biases!")
'''

