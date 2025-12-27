"""
Report generation module for DataAiPrep
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from jinja2 import Environment, FileSystemLoader, DictLoader


class ReportGenerator:
    """Generates comprehensive data quality reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # HTML template for reports
        self.html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataAiPrep - Data Quality Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007ACC;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007ACC;
            margin: 0;
            font-size: 2.5em;
        }
        .header .subtitle {
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .score-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .score {
            font-size: 4em;
            font-weight: bold;
            margin: 0;
        }
        .score-label {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section h3 {
            color: #555;
            margin-top: 25px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007ACC;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-weight: 500;
            color: #555;
        }
        .metric-value {
            font-weight: bold;
            color: #333;
        }
        .status-good {
            color: #28a745;
        }
        .status-warning {
            color: #ffc107;
        }
        .status-error {
            color: #dc3545;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .table th,
        .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #555;
        }
        .table tr:hover {
            background-color: #f5f5f5;
        }
        .recommendations {
            background: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007ACC;
        }
        .recommendations h3 {
            color: #007ACC;
            margin-top: 0;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .recommendations li {
            margin-bottom: 8px;
        }
        .issues {
            background: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }
        .issues h3 {
            color: #856404;
            margin-top: 0;
        }
        .footer {
            text-align: center;
            padding-top: 30px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>DataAiPrep</h1>
            <div class="subtitle">ML Data Quality Assessment Report</div>
            <div style="margin-top: 15px; color: #666;">
                Generated on {{ report_date }} | File: {{ file_name }}
            </div>
        </div>

        <!-- Overall Score -->
        {% if overall_score is defined %}
        <div class="score-card">
            <div class="score {{ 'status-good' if overall_score >= 80 else 'status-warning' if overall_score >= 60 else 'status-error' }}">
                {{ overall_score }}/100
            </div>
            <div class="score-label">Overall Data Quality Score</div>
        </div>
        {% endif %}

        <!-- Dataset Overview -->
        <div class="section">
            <h2>📊 Dataset Overview</h2>
            <div class="grid">
                <div class="card">
                    <h3>Basic Information</h3>
                    {% if data_info %}
                    <div class="metric">
                        <span class="metric-label">Dimensions</span>
                        <span class="metric-value">{{ data_info.shape[0]|default('N/A') }} rows × {{ data_info.shape[1]|default('N/A') }} columns</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Memory Usage</span>
                        <span class="metric-value">{{ "%.2f"|format(data_info.memory_usage|default(0) / 1024**2) }} MB</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Data Types</span>
                        <span class="metric-value">{{ data_info.dtypes|length if data_info.dtypes else 'N/A' }} different types</span>
                    </div>
                    {% endif %}
                </div>
                
                {% if completeness %}
                <div class="card">
                    <h3>Data Completeness</h3>
                    <div class="metric">
                        <span class="metric-label">Complete Columns</span>
                        <span class="metric-value status-good">{{ completeness.complete_columns_count|default(0) }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Incomplete Columns</span>
                        <span class="metric-value {{ 'status-error' if completeness.incomplete_columns_count > 0 else 'status-good' }}">{{ completeness.incomplete_columns_count|default(0) }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Overall Missing</span>
                        <span class="metric-value {{ 'status-error' if completeness.overall_missing_percentage > 15 else 'status-warning' if completeness.overall_missing_percentage > 5 else 'status-good' }}">{{ "%.2f"|format(completeness.overall_missing_percentage|default(0)) }}%</span>
                    </div>
                </div>
                {% endif %}
                
                {% if distribution %}
                <div class="card">
                    <h3>Data Distribution</h3>
                    <div class="metric">
                        <span class="metric-label">Numeric Columns</span>
                        <span class="metric-value">{{ distribution.numeric_columns|length if distribution.numeric_columns else 0 }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Categorical Columns</span>
                        <span class="metric-value">{{ distribution.categorical_columns|length if distribution.categorical_columns else 0 }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Outliers Detected</span>
                        <span class="metric-value {{ 'status-warning' if distribution.outliers|length > 0 else 'status-good' }}">{{ distribution.outliers|length if distribution.outliers else 0 }} columns</span>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- Data Completeness Details -->
        {% if completeness %}
        <div class="section">
            <h2>🔍 Data Completeness Analysis</h2>
            
            {% if completeness.missing_by_column %}
            <h3>Missing Data by Column</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Missing Count</th>
                        <th>Missing Percentage</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for column, count in completeness.missing_by_column.items() %}
                    {% set percentage = (count / completeness.total_rows * 100) if completeness.total_rows > 0 else 0 %}
                    <tr>
                        <td>{{ column }}</td>
                        <td>{{ count }}</td>
                        <td>{{ "%.2f"|format(percentage) }}%</td>
                        <td class="{{ 'status-good' if percentage == 0 else 'status-warning' if percentage < 15 else 'status-error' }}">
                            {{ 'Complete' if percentage == 0 else 'Good' if percentage < 15 else 'Poor' }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>
        {% endif %}

        <!-- Distribution Analysis -->
        {% if distribution %}
        <div class="section">
            <h2>📈 Distribution Analysis</h2>
            
            {% if distribution.statistical_summary %}
            <h3>Statistical Summary (Numeric Columns)</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Skewness</th>
                    </tr>
                </thead>
                <tbody>
                    {% for column, stats in distribution.statistical_summary.items() %}
                    <tr>
                        <td>{{ column }}</td>
                        <td>{{ "%.4f"|format(stats.mean) }}</td>
                        <td>{{ "%.4f"|format(stats.std) }}</td>
                        <td>{{ "%.4f"|format(stats.min) }}</td>
                        <td>{{ "%.4f"|format(stats.max) }}</td>
                        <td class="{{ 'status-warning' if stats.skewness|abs > 1 else 'status-good' }}">{{ "%.2f"|format(stats.skewness) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
            
            {% if distribution.outliers %}
            <h3>Outlier Detection</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>IQR Outliers</th>
                        <th>Z-Score Outliers</th>
                        <th>Modified Z-Score Outliers</th>
                    </tr>
                </thead>
                <tbody>
                    {% for column, outlier_info in distribution.outliers.items() %}
                    <tr>
                        <td>{{ column }}</td>
                        <td>{{ outlier_info.iqr.count }} ({{ "%.1f"|format(outlier_info.iqr.percentage) }}%)</td>
                        <td>{{ outlier_info.zscore.count }} ({{ "%.1f"|format(outlier_info.zscore.percentage) }}%)</td>
                        <td>{{ outlier_info.modified_zscore.count }} ({{ "%.1f"|format(outlier_info.modified_zscore.percentage) }}%)</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>
        {% endif %}

        <!-- Issues and Recommendations -->
        <div class="section">
            <h2>⚠️ Issues and Recommendations</h2>
            
            {% if all_issues %}
            <div class="issues">
                <h3>Identified Issues</h3>
                <ul>
                    {% for issue in all_issues %}
                    <li>{{ issue }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            {% if all_recommendations %}
            <div class="recommendations">
                <h3>Recommendations</h3>
                <ul>
                    {% for recommendation in all_recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Report generated by DataAiPrep - ML Data Quality Assessment Tool</p>
            <p>For more information and support, visit our documentation</p>
        </div>
    </div>
</body>
</html>
        """
        
    def generate_html_report(self, analysis_results: Dict[str, Any], 
                           output_path: Optional[str] = None,
                           file_name: str = "dataset") -> str:
        """
        Generate HTML report from analysis results
        
        Args:
            analysis_results: Results from data analysis
            output_path: Path to save the report (optional)
            file_name: Name of the analyzed file
            
        Returns:
            HTML report as string
        """
        try:
            # Prepare template data
            template_data = {
                'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'file_name': file_name,
                'data_info': analysis_results.get('data_info', {}),
                'completeness': analysis_results.get('completeness', {}),
                'distribution': analysis_results.get('distribution', {}),
                'all_issues': self._collect_all_issues(analysis_results),
                'all_recommendations': self._collect_all_recommendations(analysis_results),
                'overall_score': self._calculate_overall_score(analysis_results)
            }
            
            # Render template
            env = Environment(loader=DictLoader({'report': self.html_template}))
            template = env.get_template('report')
            html_content = template.render(**template_data)
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"HTML report saved to: {output_path}")
                
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            return f"<html><body><h1>Error generating report: {str(e)}</h1></body></html>"
            
    def generate_json_report(self, analysis_results: Dict[str, Any], 
                           output_path: Optional[str] = None) -> str:
        """
        Generate JSON report from analysis results
        
        Args:
            analysis_results: Results from data analysis
            output_path: Path to save the report (optional)
            
        Returns:
            JSON report as string
        """
        try:
            # Prepare report data
            report_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'tool_name': 'DataAiPrep',
                    'tool_version': '1.0.0'
                },
                'overall_score': self._calculate_overall_score(analysis_results),
                'summary': {
                    'total_issues': len(self._collect_all_issues(analysis_results)),
                    'total_recommendations': len(self._collect_all_recommendations(analysis_results))
                },
                'analysis_results': analysis_results
            }
            
            # Convert to JSON
            json_content = json.dumps(report_data, indent=2, default=str)
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                self.logger.info(f"JSON report saved to: {output_path}")
                
            return json_content
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {str(e)}")
            return json.dumps({'error': str(e)}, indent=2)
            
    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate executive summary text
        
        Args:
            analysis_results: Results from data analysis
            
        Returns:
            Executive summary as string
        """
        try:
            summary_lines = []
            
            # Header
            summary_lines.append("DATAAIPREP - EXECUTIVE SUMMARY")
            summary_lines.append("=" * 50)
            summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append("")
            
            # Overall score
            overall_score = self._calculate_overall_score(analysis_results)
            score_grade = self._get_score_grade(overall_score)
            summary_lines.append(f"OVERALL DATA QUALITY SCORE: {overall_score}/100 ({score_grade})")
            summary_lines.append("")
            
            # Dataset overview
            data_info = analysis_results.get('data_info', {})
            if data_info:
                shape = data_info.get('shape', [0, 0])
                memory_mb = data_info.get('memory_usage', 0) / 1024**2
                summary_lines.append("DATASET OVERVIEW:")
                summary_lines.append(f"• Dimensions: {shape[0]:,} rows × {shape[1]} columns")
                summary_lines.append(f"• Memory Usage: {memory_mb:.2f} MB")
                summary_lines.append("")
            
            # Key findings
            issues = self._collect_all_issues(analysis_results)
            summary_lines.append("KEY FINDINGS:")
            if not issues:
                summary_lines.append("• No critical issues detected")
            else:
                for issue in issues[:5]:  # Top 5 issues
                    summary_lines.append(f"• {issue}")
                if len(issues) > 5:
                    summary_lines.append(f"• ... and {len(issues) - 5} additional issues")
            summary_lines.append("")
            
            # Top recommendations
            recommendations = self._collect_all_recommendations(analysis_results)
            summary_lines.append("TOP RECOMMENDATIONS:")
            if recommendations:
                for rec in recommendations[:5]:  # Top 5 recommendations
                    summary_lines.append(f"• {rec}")
                if len(recommendations) > 5:
                    summary_lines.append(f"• ... and {len(recommendations) - 5} additional recommendations")
            else:
                summary_lines.append("• Dataset appears ready for ML training")
            summary_lines.append("")
            
            # ML readiness assessment
            summary_lines.append("ML READINESS ASSESSMENT:")
            if overall_score >= 80:
                summary_lines.append("✓ Dataset is ready for ML training")
            elif overall_score >= 60:
                summary_lines.append("⚠ Dataset requires minor improvements before ML training")
            else:
                summary_lines.append("✗ Dataset requires significant improvements before ML training")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
            
    def _collect_all_issues(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Collect all issues from analysis results"""
        all_issues = []
        
        # Completeness issues
        completeness = analysis_results.get('completeness', {})
        if completeness:
            all_issues.extend(completeness.get('issues', []))
            
        # Distribution issues
        distribution = analysis_results.get('distribution', {})
        if distribution:
            all_issues.extend(distribution.get('issues', []))
            
        return all_issues
        
    def _collect_all_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Collect all recommendations from analysis results"""
        all_recommendations = []
        
        # Completeness recommendations
        completeness = analysis_results.get('completeness', {})
        if completeness:
            all_recommendations.extend(completeness.get('recommendations', []))
            
        # Distribution recommendations
        distribution = analysis_results.get('distribution', {})
        if distribution:
            all_recommendations.extend(distribution.get('recommendations', []))
            
        return all_recommendations
        
    def _calculate_overall_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        try:
            scores = []
            weights = []
            
            # Completeness score
            completeness = analysis_results.get('completeness', {})
            if completeness:
                completeness_score = completeness.get('completeness_score', 0)
                scores.append(completeness_score)
                weights.append(0.4)  # 40% weight for completeness
                
            # Distribution score (simplified)
            distribution = analysis_results.get('distribution', {})
            if distribution:
                # Simple scoring based on issues
                issues_count = len(distribution.get('issues', []))
                distribution_score = max(0, 100 - issues_count * 10)  # -10 points per issue
                scores.append(distribution_score)
                weights.append(0.3)  # 30% weight for distribution
                
            # Base score if no specific scores available
            if not scores:
                return 75.0  # Default moderate score
                
            # Calculate weighted average
            total_weight = sum(weights)
            weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
            
            return round(weighted_score, 2)
            
        except Exception as e:
            self.logger.warning(f"Error calculating overall score: {str(e)}")
            return 0.0
            
    def _get_score_grade(self, score: float) -> str:
        """Get letter grade for score"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
            
    def export_to_pdf(self, html_content: str, output_path: str) -> bool:
        """
        Export HTML report to PDF (requires weasyprint)
        
        Args:
            html_content: HTML content to convert
            output_path: Path to save PDF
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from weasyprint import HTML
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            HTML(string=html_content).write_pdf(str(output_path))
            self.logger.info(f"PDF report saved to: {output_path}")
            return True
            
        except ImportError:
            self.logger.warning("weasyprint not available for PDF export")
            return False
        except Exception as e:
            self.logger.error(f"Error exporting to PDF: {str(e)}")
            return False