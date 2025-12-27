"""
DataQualityPipeline - Comprehensive Data Quality Assessment Pipeline

Provides a fluent API for combining multiple analysis modules
into a single, configurable pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

from .missing_value_analyzer import AdvancedMissingAnalyzer
from .outlier_detector import EnsembleOutlierDetector
from .drift_detector import DriftAnalyzer
from .fairness_analyzer import FairnessAnalyzer
from .feature_engineer import AdvancedFeatureEngineer
from .explainability import ExplainabilityEngine


@dataclass
class PipelineStep:
    """Container for a pipeline step configuration"""
    name: str
    module: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PipelineResult:
    """Container for pipeline execution results"""
    success: bool
    steps_executed: List[str]
    results: Dict[str, Any]
    errors: Dict[str, str]
    execution_time: float
    recommendations: List[Dict[str, str]]


class DataQualityPipeline:
    """
    Comprehensive data quality assessment pipeline.
    
    Combines multiple analysis modules into a single, configurable pipeline:
    - Completeness analysis (missing values, MCAR/MAR/MNAR)
    - Outlier detection (multi-method ensemble)
    - Drift detection (PSI, KS, JS divergence)
    - Fairness analysis (bias detection)
    - Feature engineering
    - Explainability analysis
    
    Example:
        pipeline = DataQualityPipeline()
        pipeline.add_step('completeness', threshold=0.95)
        pipeline.add_step('outlier_detection', methods=['iqr', 'isolation_forest'])
        pipeline.add_step('drift_detection', baseline=train_df)
        pipeline.add_step('fairness', protected_cols=['gender', 'race'])
        
        report = pipeline.run(data=df, target='label')
        report.export('quality_report.html')
    """
    
    AVAILABLE_MODULES = {
        'completeness': AdvancedMissingAnalyzer,
        'outlier_detection': EnsembleOutlierDetector,
        'drift_detection': DriftAnalyzer,
        'fairness': FairnessAnalyzer,
        'feature_engineering': AdvancedFeatureEngineer,
        'explainability': ExplainabilityEngine
    }
    
    def __init__(self, name: str = "DataQualityPipeline"):
        """
        Initialize the pipeline.
        
        Args:
            name: Name for this pipeline instance
        """
        self.name = name
        self.steps: List[PipelineStep] = []
        self.results_: Optional[PipelineResult] = None
        self._baseline_data: Optional[pd.DataFrame] = None
        self._protected_columns: List[str] = []
        
    def add_step(
        self,
        module: str,
        name: str = None,
        **config
    ) -> 'DataQualityPipeline':
        """
        Add a step to the pipeline.
        
        Args:
            module: Module name ('completeness', 'outlier_detection', etc.)
            name: Optional custom name for this step
            **config: Module-specific configuration
            
        Returns:
            Self for method chaining
        """
        if module not in self.AVAILABLE_MODULES:
            raise ValueError(f"Unknown module: {module}. Available: {list(self.AVAILABLE_MODULES.keys())}")
        
        step = PipelineStep(
            name=name or f"{module}_{len(self.steps)}",
            module=module,
            config=config
        )
        
        # Store special config
        if 'baseline' in config:
            self._baseline_data = config.pop('baseline')
        if 'protected_cols' in config:
            self._protected_columns = config.pop('protected_cols')
        
        self.steps.append(step)
        return self
    
    def remove_step(self, name: str) -> 'DataQualityPipeline':
        """Remove a step by name"""
        self.steps = [s for s in self.steps if s.name != name]
        return self
    
    def enable_step(self, name: str) -> 'DataQualityPipeline':
        """Enable a step by name"""
        for step in self.steps:
            if step.name == name:
                step.enabled = True
        return self
    
    def disable_step(self, name: str) -> 'DataQualityPipeline':
        """Disable a step by name"""
        for step in self.steps:
            if step.name == name:
                step.enabled = False
        return self
    
    def run(
        self,
        data: pd.DataFrame,
        target: str = None,
        verbose: bool = True
    ) -> 'DataQualityPipeline':
        """
        Execute the pipeline.
        
        Args:
            data: DataFrame to analyze
            target: Target column name
            verbose: Whether to print progress
            
        Returns:
            Self with results available via .results_
        """
        start_time = datetime.now()
        
        results = {}
        errors = {}
        steps_executed = []
        all_recommendations = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"EXECUTING PIPELINE: {self.name}")
            print(f"{'='*60}")
            print(f"Data shape: {data.shape}")
            print(f"Steps configured: {len(self.steps)}")
            print(f"Target column: {target or 'Not specified'}")
            print()
        
        for step in self.steps:
            if not step.enabled:
                if verbose:
                    print(f"⏭️  Skipping {step.name} (disabled)")
                continue
            
            if verbose:
                print(f"▶️  Running {step.name}...", end=" ")
            
            try:
                step_result = self._execute_step(step, data, target)
                results[step.name] = step_result
                steps_executed.append(step.name)
                
                # Extract recommendations
                if isinstance(step_result, dict):
                    if 'recommendations' in step_result:
                        all_recommendations.extend(step_result['recommendations'])
                    if 'alerts' in step_result:
                        for alert in step_result['alerts']:
                            all_recommendations.append({
                                'category': step.module,
                                'severity': alert.get('severity', 'medium'),
                                'issue': alert.get('message', str(alert))
                            })
                
                if verbose:
                    print("✅")
                    
            except Exception as e:
                errors[step.name] = str(e)
                if verbose:
                    print(f"❌ Error: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.results_ = PipelineResult(
            success=len(errors) == 0,
            steps_executed=steps_executed,
            results=results,
            errors=errors,
            execution_time=execution_time,
            recommendations=all_recommendations
        )
        
        if verbose:
            print()
            print(f"{'='*60}")
            print(f"PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Steps executed: {len(steps_executed)}/{len(self.steps)}")
            print(f"Errors: {len(errors)}")
            print(f"Execution time: {execution_time:.2f}s")
            print(f"Recommendations: {len(all_recommendations)}")
        
        return self
    
    def _execute_step(
        self,
        step: PipelineStep,
        data: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """Execute a single pipeline step"""
        module_class = self.AVAILABLE_MODULES[step.module]
        
        if step.module == 'completeness':
            analyzer = module_class(**step.config)
            return analyzer.analyze(data)
        
        elif step.module == 'outlier_detection':
            detector = module_class(**step.config)
            return detector.detect(data)
        
        elif step.module == 'drift_detection':
            if self._baseline_data is None:
                return {'error': 'No baseline data provided for drift detection'}
            analyzer = module_class(**step.config)
            return analyzer.analyze(self._baseline_data, data, target)
        
        elif step.module == 'fairness':
            analyzer = module_class(
                protected_columns=self._protected_columns,
                **step.config
            )
            if target:
                return analyzer.analyze(data, target)
            else:
                return {'error': 'Target column required for fairness analysis'}
        
        elif step.module == 'feature_engineering':
            engineer = module_class(**step.config)
            engineered_df, report = engineer.engineer_features(
                data, target_column=target
            )
            return report
        
        elif step.module == 'explainability':
            engine = module_class(**step.config)
            if target:
                return engine.analyze(data, target)
            else:
                return {'error': 'Target column required for explainability analysis'}
        
        return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        if self.results_ is None:
            return {'error': 'Pipeline has not been executed yet'}
        
        return {
            'pipeline_name': self.name,
            'success': self.results_.success,
            'steps_executed': self.results_.steps_executed,
            'errors': self.results_.errors,
            'execution_time': self.results_.execution_time,
            'total_recommendations': len(self.results_.recommendations),
            'high_severity_issues': len([
                r for r in self.results_.recommendations 
                if r.get('severity') == 'high'
            ])
        }
    
    def get_recommendations(
        self,
        severity: str = None
    ) -> List[Dict[str, str]]:
        """Get recommendations, optionally filtered by severity"""
        if self.results_ is None:
            return []
        
        recommendations = self.results_.recommendations
        
        if severity:
            recommendations = [
                r for r in recommendations 
                if r.get('severity') == severity
            ]
        
        return recommendations
    
    def export(
        self,
        path: str,
        format: str = 'html'
    ) -> str:
        """
        Export pipeline results.
        
        Args:
            path: Output file path
            format: Output format ('html', 'json', 'text')
            
        Returns:
            Path to exported file
        """
        if self.results_ is None:
            raise ValueError("Pipeline has not been executed yet")
        
        if format == 'json':
            return self._export_json(path)
        elif format == 'html':
            return self._export_html(path)
        elif format == 'text':
            return self._export_text(path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_json(self, path: str) -> str:
        """Export results as JSON"""
        # Convert results to JSON-serializable format
        output = {
            'pipeline_name': self.name,
            'execution_time': self.results_.execution_time,
            'success': self.results_.success,
            'steps_executed': self.results_.steps_executed,
            'errors': self.results_.errors,
            'recommendations': self.results_.recommendations,
            'results': self._serialize_results(self.results_.results)
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        return path
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Convert results to JSON-serializable format"""
        serialized = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_results(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = [
                    self._serialize_results(v) if isinstance(v, dict) else v
                    for v in value
                ]
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                serialized[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                serialized[key] = value.to_dict()
            else:
                serialized[key] = value
        
        return serialized
    
    def _export_html(self, path: str) -> str:
        """Export results as HTML report"""
        html = self._generate_html_report()
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return path
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataAiPrep Quality Report - {self.name}</title>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --text-primary: #e8e8f0;
            --text-secondary: #a0a0b0;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #2a2a3a;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 12px;
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: rgba(255,255,255,0.8);
            font-size: 1.1rem;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            text-align: center;
        }}
        
        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-primary);
        }}
        
        .summary-card .label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        
        .section {{
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}
        
        .section-header {{
            background: var(--bg-secondary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
        }}
        
        .section-content {{
            padding: 1.5rem;
        }}
        
        .recommendation {{
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 6px;
            border-left: 4px solid;
        }}
        
        .recommendation.high {{
            background: rgba(239, 68, 68, 0.1);
            border-color: var(--error);
        }}
        
        .recommendation.medium {{
            background: rgba(245, 158, 11, 0.1);
            border-color: var(--warning);
        }}
        
        .recommendation.low {{
            background: rgba(34, 197, 94, 0.1);
            border-color: var(--success);
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge.success {{ background: var(--success); color: white; }}
        .badge.error {{ background: var(--error); color: white; }}
        .badge.warning {{ background: var(--warning); color: black; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background: var(--bg-secondary);
            font-weight: 600;
        }}
        
        .status-success {{ color: var(--success); }}
        .status-error {{ color: var(--error); }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 DataAiPrep Quality Report</h1>
            <p class="subtitle">{self.name}</p>
        </header>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{len(self.results_.steps_executed)}</div>
                <div class="label">Steps Executed</div>
            </div>
            <div class="summary-card">
                <div class="value">{self.results_.execution_time:.2f}s</div>
                <div class="label">Execution Time</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(self.results_.recommendations)}</div>
                <div class="label">Recommendations</div>
            </div>
            <div class="summary-card">
                <div class="value">{'✅' if self.results_.success else '❌'}</div>
                <div class="label">Status</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">Pipeline Steps</div>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>Step</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''
                        <tr>
                            <td>{step.name}</td>
                            <td class="{'status-success' if step.name in self.results_.steps_executed else 'status-error'}">
                                {'✅ Completed' if step.name in self.results_.steps_executed else '❌ ' + self.results_.errors.get(step.name, 'Skipped')}
                            </td>
                        </tr>
                        ''' for step in self.steps)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">Recommendations</div>
            <div class="section-content">
                {''.join(f'''
                <div class="recommendation {rec.get('severity', 'medium')}">
                    <span class="badge {rec.get('severity', 'medium')}">{rec.get('severity', 'medium')}</span>
                    <strong> {rec.get('category', 'General')}</strong><br>
                    {rec.get('issue', rec.get('message', str(rec)))}
                </div>
                ''' for rec in self.results_.recommendations[:20]) or '<p>No recommendations generated.</p>'}
            </div>
        </div>
        
        <footer>
            Generated by DataAiPrep Quality Pipeline<br>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </footer>
    </div>
</body>
</html>
"""
        return html
    
    def _export_text(self, path: str) -> str:
        """Export results as text report"""
        report = []
        report.append("=" * 60)
        report.append(f"DATAAIPREP QUALITY REPORT: {self.name}")
        report.append("=" * 60)
        report.append("")
        
        summary = self.get_summary()
        report.append(f"Status: {'SUCCESS' if summary['success'] else 'FAILED'}")
        report.append(f"Execution Time: {summary['execution_time']:.2f}s")
        report.append(f"Steps Executed: {len(summary['steps_executed'])}")
        report.append(f"Total Recommendations: {summary['total_recommendations']}")
        report.append(f"High Severity Issues: {summary['high_severity_issues']}")
        report.append("")
        
        report.append("-" * 60)
        report.append("RECOMMENDATIONS")
        report.append("-" * 60)
        
        for rec in self.results_.recommendations[:20]:
            severity = rec.get('severity', 'medium').upper()
            category = rec.get('category', 'General')
            issue = rec.get('issue', rec.get('message', str(rec)))
            report.append(f"\n[{severity}] {category}")
            report.append(f"  {issue}")
        
        text = "\n".join(report)
        
        with open(path, 'w') as f:
            f.write(text)
        
        return path
    
    def __repr__(self) -> str:
        return f"DataQualityPipeline(name='{self.name}', steps={len(self.steps)})"


# Convenience function for quick analysis
def quick_analyze(
    data: pd.DataFrame,
    target: str = None,
    baseline: pd.DataFrame = None,
    protected_cols: List[str] = None
) -> DataQualityPipeline:
    """
    Quick analysis with sensible defaults.
    
    Args:
        data: DataFrame to analyze
        target: Target column name
        baseline: Baseline data for drift detection
        protected_cols: Protected attribute columns
        
    Returns:
        Executed pipeline with results
    """
    pipeline = DataQualityPipeline(name="QuickAnalysis")
    
    # Add default steps
    pipeline.add_step('completeness')
    pipeline.add_step('outlier_detection', methods=['iqr', 'zscore', 'isolation_forest'])
    
    if baseline is not None:
        pipeline.add_step('drift_detection', baseline=baseline)
    
    if protected_cols and target:
        pipeline.add_step('fairness', protected_cols=protected_cols)
    
    if target:
        pipeline.add_step('explainability')
    
    return pipeline.run(data, target=target)

