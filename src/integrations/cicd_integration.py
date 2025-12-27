"""
CI/CD Integration for DataAiPrep

Provides utilities for integrating data quality checks into CI/CD pipelines.

Supports:
- GitHub Actions
- GitLab CI
- Jenkins
- Azure DevOps
- Generic CI/CD
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd


class DataAiPrepCI:
    """
    CI/CD integration utility for DataAiPrep.
    
    Features:
    - Command-line interface for CI/CD scripts
    - Exit codes for pipeline integration
    - JSON output for downstream processing
    - Quality gates with configurable thresholds
    """
    
    def __init__(
        self,
        completeness_threshold: float = 95.0,
        leakage_threshold: float = 30.0,
        missing_threshold: float = 10.0,
        output_format: str = 'json'
    ):
        """
        Initialize CI integration.
        
        Args:
            completeness_threshold: Minimum completeness score (0-100)
            leakage_threshold: Maximum allowed leakage score (0-100)
            missing_threshold: Maximum allowed missing percentage (0-100)
            output_format: Output format ('json', 'text', 'github')
        """
        self.completeness_threshold = completeness_threshold
        self.leakage_threshold = leakage_threshold
        self.missing_threshold = missing_threshold
        self.output_format = output_format
        
    def check_data_quality(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run data quality check and return results.
        
        Args:
            data_path: Path to data file
            target_column: Optional target column name
            output_path: Optional path to save results
            
        Returns:
            Dictionary with results and pass/fail status
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Import analyzers
        from ..analysis.completeness_analyzer import CompletenessAnalyzer
        from ..analysis.distribution_analyzer import DistributionAnalyzer
        from ..analysis.leakage_detector import LeakageDetector
        
        # Run assessments
        completeness = CompletenessAnalyzer().analyze(df)
        distribution = DistributionAnalyzer().analyze(df, target_column)
        leakage = LeakageDetector().analyze(df, target_column)
        
        # Extract key metrics
        completeness_score = completeness.get('completeness_score', 0)
        missing_pct = completeness.get('overall_missing_percentage', 0)
        leakage_score = leakage.get('leakage_score', 0)
        
        # Determine pass/fail
        issues = []
        passed = True
        
        if completeness_score < self.completeness_threshold:
            issues.append(f"Completeness {completeness_score:.1f}% < {self.completeness_threshold}%")
            passed = False
            
        if missing_pct > self.missing_threshold:
            issues.append(f"Missing {missing_pct:.1f}% > {self.missing_threshold}%")
            passed = False
            
        if leakage_score > self.leakage_threshold:
            issues.append(f"Leakage score {leakage_score:.1f} > {self.leakage_threshold}")
            passed = False
        
        results = {
            'passed': passed,
            'data_file': data_path,
            'rows': len(df),
            'columns': len(df.columns),
            'metrics': {
                'completeness_score': completeness_score,
                'missing_percentage': missing_pct,
                'leakage_score': leakage_score,
                'perfect_correlations': len(leakage.get('perfect_correlations', [])),
                'duplicate_features': len(leakage.get('duplicate_features', []))
            },
            'thresholds': {
                'completeness_min': self.completeness_threshold,
                'missing_max': self.missing_threshold,
                'leakage_max': self.leakage_threshold
            },
            'issues': issues,
            'recommendations': completeness.get('recommendations', []) + 
                              leakage.get('recommendations', [])
        }
        
        # Save results if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def format_output(self, results: Dict[str, Any]) -> str:
        """
        Format results for CI/CD output.
        
        Args:
            results: Results dictionary from check_data_quality
            
        Returns:
            Formatted string output
        """
        if self.output_format == 'json':
            return json.dumps(results, indent=2)
        
        elif self.output_format == 'github':
            # GitHub Actions workflow commands
            output = []
            
            status = "✅ PASSED" if results['passed'] else "❌ FAILED"
            output.append(f"::{'notice' if results['passed'] else 'error'}::Data Quality Check {status}")
            
            # Set output variables
            output.append(f"::set-output name=passed::{str(results['passed']).lower()}")
            output.append(f"::set-output name=completeness::{results['metrics']['completeness_score']}")
            output.append(f"::set-output name=leakage::{results['metrics']['leakage_score']}")
            
            # Log issues as warnings
            for issue in results['issues']:
                output.append(f"::warning::{issue}")
            
            return '\n'.join(output)
        
        else:  # text format
            output = []
            output.append("=" * 60)
            output.append("DataAiPrep Data Quality Check")
            output.append("=" * 60)
            output.append(f"Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
            output.append(f"Data: {results['data_file']}")
            output.append(f"Shape: {results['rows']} rows × {results['columns']} columns")
            output.append("")
            output.append("Metrics:")
            output.append(f"  Completeness Score: {results['metrics']['completeness_score']:.1f}%")
            output.append(f"  Missing Percentage: {results['metrics']['missing_percentage']:.1f}%")
            output.append(f"  Leakage Risk Score: {results['metrics']['leakage_score']:.1f}/100")
            output.append(f"  Perfect Correlations: {results['metrics']['perfect_correlations']}")
            output.append(f"  Duplicate Features: {results['metrics']['duplicate_features']}")
            
            if results['issues']:
                output.append("")
                output.append("Issues Found:")
                for issue in results['issues']:
                    output.append(f"  ❌ {issue}")
            
            output.append("=" * 60)
            return '\n'.join(output)
    
    def get_exit_code(self, results: Dict[str, Any]) -> int:
        """
        Get appropriate exit code for CI/CD.
        
        Args:
            results: Results dictionary
            
        Returns:
            0 if passed, 1 if failed
        """
        return 0 if results['passed'] else 1


def main():
    """Command-line interface for CI/CD integration"""
    parser = argparse.ArgumentParser(
        description='DataAiPrep Data Quality Check for CI/CD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.integrations.cicd_integration data.csv
  python -m src.integrations.cicd_integration data.csv --target target_column
  python -m src.integrations.cicd_integration data.csv --format github --output results.json
        """
    )
    
    parser.add_argument('data_file', help='Path to data file (CSV)')
    parser.add_argument('--target', '-t', help='Target column name')
    parser.add_argument('--output', '-o', help='Output file path for results')
    parser.add_argument('--format', '-f', choices=['json', 'text', 'github'], 
                       default='text', help='Output format')
    parser.add_argument('--completeness-threshold', type=float, default=95.0,
                       help='Minimum completeness score (default: 95)')
    parser.add_argument('--missing-threshold', type=float, default=10.0,
                       help='Maximum missing percentage (default: 10)')
    parser.add_argument('--leakage-threshold', type=float, default=30.0,
                       help='Maximum leakage score (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize CI integration
    ci = DataAiPrepCI(
        completeness_threshold=args.completeness_threshold,
        leakage_threshold=args.leakage_threshold,
        missing_threshold=args.missing_threshold,
        output_format=args.format
    )
    
    # Run quality check
    results = ci.check_data_quality(
        data_path=args.data_file,
        target_column=args.target,
        output_path=args.output
    )
    
    # Print formatted output
    print(ci.format_output(results))
    
    # Exit with appropriate code
    sys.exit(ci.get_exit_code(results))


# CI/CD Configuration Examples
GITHUB_ACTIONS_EXAMPLE = '''
# .github/workflows/data-quality.yml
# ==================================

name: Data Quality Check

on:
  push:
    paths:
      - 'data/**'
  pull_request:
    paths:
      - 'data/**'

jobs:
  quality-check:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install pandas numpy scikit-learn scipy
          pip install -e .  # Install DataAiPrep
      
      - name: Run Data Quality Check
        id: quality
        run: |
          python -m src.integrations.cicd_integration \\
            data/training_data.csv \\
            --target target \\
            --format github \\
            --output quality_report.json
      
      - name: Upload Quality Report
        uses: actions/upload-artifact@v3
        with:
          name: data-quality-report
          path: quality_report.json
      
      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('quality_report.json', 'utf8'));
            const status = report.passed ? '✅' : '❌';
            const body = `## Data Quality Check ${status}
            
            | Metric | Value | Threshold |
            |--------|-------|-----------|
            | Completeness | ${report.metrics.completeness_score.toFixed(1)}% | ≥${report.thresholds.completeness_min}% |
            | Missing Data | ${report.metrics.missing_percentage.toFixed(1)}% | ≤${report.thresholds.missing_max}% |
            | Leakage Risk | ${report.metrics.leakage_score.toFixed(1)} | ≤${report.thresholds.leakage_max} |
            `;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
'''

GITLAB_CI_EXAMPLE = '''
# .gitlab-ci.yml
# ==============

stages:
  - quality
  - train

data-quality:
  stage: quality
  image: python:3.9
  script:
    - pip install pandas numpy scikit-learn scipy
    - pip install -e .
    - python -m src.integrations.cicd_integration 
        data/training_data.csv 
        --target target 
        --format text 
        --output quality_report.json
  artifacts:
    paths:
      - quality_report.json
    reports:
      dotenv: quality.env
  rules:
    - changes:
        - data/**

train-model:
  stage: train
  needs: [data-quality]
  script:
    - echo "Training model with quality-checked data"
'''

JENKINS_EXAMPLE = '''
// Jenkinsfile
// ===========

pipeline {
    agent any
    
    stages {
        stage('Data Quality Check') {
            steps {
                sh '''
                    pip install pandas numpy scikit-learn scipy
                    pip install -e .
                    python -m src.integrations.cicd_integration \\
                        data/training_data.csv \\
                        --target target \\
                        --format text \\
                        --output quality_report.json
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'quality_report.json'
                }
            }
        }
        
        stage('Train Model') {
            when {
                expression {
                    def report = readJSON file: 'quality_report.json'
                    return report.passed
                }
            }
            steps {
                sh 'python train_model.py'
            }
        }
    }
}
'''


if __name__ == "__main__":
    main()

