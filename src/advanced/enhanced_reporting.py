"""
Enhanced Reporting Module

Features:
- Interactive HTML Reports with Plotly
- PDF Export
- Model Cards Generation
- Data Sheets Documentation
- Executive Dashboards
- Multiple Output Formats (HTML, PDF, JSON, Markdown)
"""

import json
import base64
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ReportSection:
    """Container for a report section"""
    title: str
    content_type: str  # 'text', 'table', 'chart', 'metric', 'alert'
    content: Any
    priority: int = 0  # Higher = more important


class InteractiveReportGenerator:
    """
    Generate interactive HTML reports with Plotly visualizations.
    
    Features:
    - Interactive charts and graphs
    - Drill-down tables
    - Responsive design
    - Dark/light theme support
    - Export to multiple formats
    """
    
    def __init__(
        self,
        title: str = "DataAiPrep Quality Report",
        theme: str = "dark",
        include_plotly: bool = True
    ):
        """
        Initialize the report generator.
        
        Args:
            title: Report title
            theme: Color theme ('dark' or 'light')
            include_plotly: Whether to include Plotly.js
        """
        self.title = title
        self.theme = theme
        self.include_plotly = include_plotly
        self._plotly_available = self._check_plotly()
        self._sections: List[ReportSection] = []
        
    def _check_plotly(self) -> bool:
        """Check if Plotly is available"""
        try:
            import plotly
            import plotly.graph_objects as go
            import plotly.express as px
            self._plotly = plotly
            self._go = go
            self._px = px
            return True
        except ImportError:
            return False
    
    def add_section(
        self,
        title: str,
        content_type: str,
        content: Any,
        priority: int = 0
    ):
        """Add a section to the report"""
        self._sections.append(ReportSection(
            title=title,
            content_type=content_type,
            content=content,
            priority=priority
        ))
    
    def add_summary_metrics(
        self,
        metrics: Dict[str, Union[float, int, str]],
        title: str = "Summary Metrics"
    ):
        """Add summary metrics cards"""
        self.add_section(title, 'metric', metrics, priority=10)
    
    def add_quality_scores(
        self,
        scores: Dict[str, float],
        title: str = "Quality Scores"
    ):
        """Add quality score gauge charts"""
        self.add_section(title, 'gauge', scores, priority=9)
    
    def add_distribution_chart(
        self,
        data: Dict[str, List[float]],
        title: str = "Distribution Analysis"
    ):
        """Add distribution histogram/box plot"""
        self.add_section(title, 'distribution', data, priority=7)
    
    def add_correlation_heatmap(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        title: str = "Feature Correlations"
    ):
        """Add correlation heatmap"""
        self.add_section(title, 'heatmap', correlation_matrix, priority=6)
    
    def add_feature_importance(
        self,
        importance: Dict[str, float],
        title: str = "Feature Importance"
    ):
        """Add feature importance bar chart"""
        self.add_section(title, 'bar', importance, priority=8)
    
    def add_table(
        self,
        data: List[Dict[str, Any]],
        title: str = "Data Table",
        sortable: bool = True
    ):
        """Add an interactive data table"""
        self.add_section(title, 'table', {'data': data, 'sortable': sortable}, priority=5)
    
    def add_alerts(
        self,
        alerts: List[Dict[str, Any]],
        title: str = "Quality Alerts"
    ):
        """Add alerts/recommendations section"""
        self.add_section(title, 'alert', alerts, priority=10)
    
    def add_text(
        self,
        text: str,
        title: str = "",
        format: str = "markdown"
    ):
        """Add text content"""
        self.add_section(title, 'text', {'text': text, 'format': format}, priority=3)
    
    def generate_html(self) -> str:
        """Generate the complete HTML report"""
        # Sort sections by priority
        sections = sorted(self._sections, key=lambda x: -x.priority)
        
        html_parts = [self._get_html_header()]
        
        # Generate each section
        for section in sections:
            html_parts.append(self._render_section(section))
        
        html_parts.append(self._get_html_footer())
        
        return '\n'.join(html_parts)
    
    def _get_html_header(self) -> str:
        """Get HTML header with styles"""
        colors = self._get_theme_colors()
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    {'<script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>' if self.include_plotly else ''}
    <style>
        :root {{
            --bg-primary: {colors['bg_primary']};
            --bg-secondary: {colors['bg_secondary']};
            --bg-card: {colors['bg_card']};
            --text-primary: {colors['text_primary']};
            --text-secondary: {colors['text_secondary']};
            --accent: {colors['accent']};
            --accent-secondary: {colors['accent_secondary']};
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --critical: #dc2626;
            --border: {colors['border']};
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: rgba(255,255,255,0.85);
            font-size: 1.1rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }}
        
        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.5rem;
        }}
        
        .metric-card .label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .section {{
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
            overflow: hidden;
        }}
        
        .section-header {{
            background: var(--bg-secondary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 1.1rem;
        }}
        
        .section-content {{
            padding: 1.5rem;
        }}
        
        .alert {{
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            border-left: 4px solid;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        }}
        
        .alert.critical {{ background: rgba(220, 38, 38, 0.15); border-color: var(--critical); }}
        .alert.error {{ background: rgba(239, 68, 68, 0.15); border-color: var(--error); }}
        .alert.warning {{ background: rgba(245, 158, 11, 0.15); border-color: var(--warning); }}
        .alert.info {{ background: rgba(59, 130, 246, 0.15); border-color: #3b82f6; }}
        
        .alert-icon {{ font-size: 1.5rem; }}
        .alert-content {{ flex: 1; }}
        .alert-title {{ font-weight: 600; margin-bottom: 0.25rem; }}
        .alert-message {{ color: var(--text-secondary); font-size: 0.9rem; }}
        
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
            cursor: pointer;
        }}
        
        th:hover {{
            background: var(--bg-primary);
        }}
        
        tr:hover {{
            background: var(--bg-secondary);
        }}
        
        .chart-container {{
            min-height: 400px;
        }}
        
        .gauge-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }}
        
        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .bar-item {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .bar-label {{
            width: 150px;
            font-size: 0.9rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .bar-track {{
            flex: 1;
            height: 24px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--accent-secondary));
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .bar-value {{
            width: 60px;
            text-align: right;
            font-weight: 500;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            header {{ padding: 2rem 1rem; }}
            h1 {{ font-size: 1.8rem; }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {self.title}</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""
    
    def _get_theme_colors(self) -> Dict[str, str]:
        """Get color scheme for current theme"""
        if self.theme == 'dark':
            return {
                'bg_primary': '#0a0a0f',
                'bg_secondary': '#12121a',
                'bg_card': '#1a1a24',
                'text_primary': '#e8e8f0',
                'text_secondary': '#a0a0b0',
                'accent': '#6366f1',
                'accent_secondary': '#8b5cf6',
                'border': '#2a2a3a'
            }
        else:
            return {
                'bg_primary': '#f5f5f5',
                'bg_secondary': '#ffffff',
                'bg_card': '#ffffff',
                'text_primary': '#1a1a2e',
                'text_secondary': '#6b6b80',
                'accent': '#4f46e5',
                'accent_secondary': '#7c3aed',
                'border': '#e0e0e0'
            }
    
    def _render_section(self, section: ReportSection) -> str:
        """Render a single section to HTML"""
        if section.content_type == 'metric':
            return self._render_metrics(section)
        elif section.content_type == 'gauge':
            return self._render_gauges(section)
        elif section.content_type == 'alert':
            return self._render_alerts(section)
        elif section.content_type == 'table':
            return self._render_table(section)
        elif section.content_type == 'bar':
            return self._render_bar_chart(section)
        elif section.content_type == 'distribution':
            return self._render_distribution(section)
        elif section.content_type == 'heatmap':
            return self._render_heatmap(section)
        elif section.content_type == 'text':
            return self._render_text(section)
        else:
            return ''
    
    def _render_metrics(self, section: ReportSection) -> str:
        """Render metric cards"""
        metrics = section.content
        cards = []
        
        for label, value in metrics.items():
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            cards.append(f"""
            <div class="metric-card">
                <div class="value">{display_value}</div>
                <div class="label">{label}</div>
            </div>
            """)
        
        return f"""
        <div class="metrics-grid">
            {''.join(cards)}
        </div>
        """
    
    def _render_gauges(self, section: ReportSection) -> str:
        """Render gauge charts for scores"""
        scores = section.content
        
        if self._plotly_available:
            gauges = []
            for i, (label, value) in enumerate(scores.items()):
                gauge_id = f"gauge_{i}"
                color = '#22c55e' if value >= 80 else ('#f59e0b' if value >= 60 else '#ef4444')
                
                gauges.append(f"""
                <div id="{gauge_id}" style="height: 200px;"></div>
                <script>
                    Plotly.newPlot('{gauge_id}', [{{
                        type: 'indicator',
                        mode: 'gauge+number',
                        value: {value},
                        title: {{ text: '{label}', font: {{ color: 'var(--text-primary)' }} }},
                        gauge: {{
                            axis: {{ range: [0, 100], tickcolor: 'var(--text-secondary)' }},
                            bar: {{ color: '{color}' }},
                            bgcolor: 'var(--bg-secondary)',
                            bordercolor: 'var(--border)'
                        }}
                    }}], {{
                        paper_bgcolor: 'transparent',
                        font: {{ color: 'var(--text-primary)' }},
                        margin: {{ t: 80, b: 20, l: 30, r: 30 }}
                    }}, {{ responsive: true }});
                </script>
                """)
            
            return f"""
            <div class="section">
                <div class="section-header">{section.title}</div>
                <div class="section-content">
                    <div class="gauge-grid">
                        {''.join(gauges)}
                    </div>
                </div>
            </div>
            """
        else:
            # Fallback to simple bars
            return self._render_bar_chart(section)
    
    def _render_alerts(self, section: ReportSection) -> str:
        """Render alerts section"""
        alerts = section.content
        icons = {
            'critical': '🚨',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        alert_html = []
        for alert in alerts:
            severity = alert.get('severity', 'info')
            icon = icons.get(severity, 'ℹ️')
            
            alert_html.append(f"""
            <div class="alert {severity}">
                <span class="alert-icon">{icon}</span>
                <div class="alert-content">
                    <div class="alert-title">{alert.get('title', alert.get('category', ''))}</div>
                    <div class="alert-message">{alert.get('message', alert.get('issue', ''))}</div>
                </div>
            </div>
            """)
        
        return f"""
        <div class="section">
            <div class="section-header">{section.title}</div>
            <div class="section-content">
                {''.join(alert_html) if alert_html else '<p>No alerts to display.</p>'}
            </div>
        </div>
        """
    
    def _render_table(self, section: ReportSection) -> str:
        """Render interactive table"""
        data = section.content.get('data', [])
        
        if not data:
            return ''
        
        # Get columns from first row
        columns = list(data[0].keys())
        
        headers = ''.join(f'<th onclick="sortTable(this)">{col}</th>' for col in columns)
        
        rows = []
        for row in data:
            cells = ''.join(f'<td>{row.get(col, "")}</td>' for col in columns)
            rows.append(f'<tr>{cells}</tr>')
        
        return f"""
        <div class="section">
            <div class="section-header">{section.title}</div>
            <div class="section-content">
                <table>
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        </div>
        <script>
            function sortTable(th) {{
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const idx = Array.from(th.parentNode.children).indexOf(th);
                const asc = th.dataset.sort !== 'asc';
                
                rows.sort((a, b) => {{
                    const aVal = a.children[idx].textContent;
                    const bVal = b.children[idx].textContent;
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return asc ? aNum - bNum : bNum - aNum;
                    }}
                    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }});
                
                th.dataset.sort = asc ? 'asc' : 'desc';
                rows.forEach(row => tbody.appendChild(row));
            }}
        </script>
        """
    
    def _render_bar_chart(self, section: ReportSection) -> str:
        """Render horizontal bar chart"""
        data = section.content
        
        if self._plotly_available and len(data) > 0:
            chart_id = f"bar_{id(section)}"
            labels = list(data.keys())[:20]  # Top 20
            values = [data[k] for k in labels]
            
            return f"""
            <div class="section">
                <div class="section-header">{section.title}</div>
                <div class="section-content">
                    <div id="{chart_id}" class="chart-container"></div>
                </div>
            </div>
            <script>
                Plotly.newPlot('{chart_id}', [{{
                    type: 'bar',
                    x: {json.dumps(values)},
                    y: {json.dumps(labels)},
                    orientation: 'h',
                    marker: {{
                        color: {json.dumps(values)},
                        colorscale: 'Viridis'
                    }}
                }}], {{
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: {{ color: 'var(--text-primary)' }},
                    xaxis: {{ gridcolor: 'var(--border)' }},
                    yaxis: {{ gridcolor: 'var(--border)', autorange: 'reversed' }},
                    margin: {{ l: 150, r: 30, t: 30, b: 50 }}
                }}, {{ responsive: true }});
            </script>
            """
        
        # Fallback to CSS bars
        max_val = max(data.values()) if data else 1
        bars = []
        for label, value in list(data.items())[:20]:
            pct = (value / max_val) * 100
            bars.append(f"""
            <div class="bar-item">
                <span class="bar-label" title="{label}">{label}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {pct}%"></div>
                </div>
                <span class="bar-value">{value:.4f}</span>
            </div>
            """)
        
        return f"""
        <div class="section">
            <div class="section-header">{section.title}</div>
            <div class="section-content">
                <div class="bar-chart">
                    {''.join(bars)}
                </div>
            </div>
        </div>
        """
    
    def _render_distribution(self, section: ReportSection) -> str:
        """Render distribution charts"""
        if not self._plotly_available:
            return self._render_text(ReportSection(
                title=section.title,
                content_type='text',
                content={'text': 'Distribution charts require Plotly library.', 'format': 'text'}
            ))
        
        data = section.content
        chart_id = f"dist_{id(section)}"
        
        traces = []
        for col, values in data.items():
            traces.append(f"{{ type: 'histogram', x: {json.dumps(values)}, name: '{col}', opacity: 0.7 }}")
        
        return f"""
        <div class="section">
            <div class="section-header">{section.title}</div>
            <div class="section-content">
                <div id="{chart_id}" class="chart-container"></div>
            </div>
        </div>
        <script>
            Plotly.newPlot('{chart_id}', [{','.join(traces)}], {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {{ color: 'var(--text-primary)' }},
                xaxis: {{ gridcolor: 'var(--border)' }},
                yaxis: {{ gridcolor: 'var(--border)' }},
                barmode: 'overlay',
                margin: {{ t: 30, b: 50 }}
            }}, {{ responsive: true }});
        </script>
        """
    
    def _render_heatmap(self, section: ReportSection) -> str:
        """Render correlation heatmap"""
        if not self._plotly_available:
            return ''
        
        data = section.content
        chart_id = f"heatmap_{id(section)}"
        
        labels = list(data.keys())
        z_values = [[data[row].get(col, 0) for col in labels] for row in labels]
        
        return f"""
        <div class="section">
            <div class="section-header">{section.title}</div>
            <div class="section-content">
                <div id="{chart_id}" class="chart-container"></div>
            </div>
        </div>
        <script>
            Plotly.newPlot('{chart_id}', [{{
                type: 'heatmap',
                z: {json.dumps(z_values)},
                x: {json.dumps(labels)},
                y: {json.dumps(labels)},
                colorscale: 'RdBu',
                zmid: 0
            }}], {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {{ color: 'var(--text-primary)' }},
                margin: {{ t: 30, b: 100, l: 100 }}
            }}, {{ responsive: true }});
        </script>
        """
    
    def _render_text(self, section: ReportSection) -> str:
        """Render text content"""
        text = section.content.get('text', '')
        
        return f"""
        <div class="section">
            {f'<div class="section-header">{section.title}</div>' if section.title else ''}
            <div class="section-content">
                <p>{text}</p>
            </div>
        </div>
        """
    
    def _get_html_footer(self) -> str:
        """Get HTML footer"""
        return """
        <footer>
            <p>Generated by DataAiPrep - Advanced ML Data Quality Assessment</p>
            <p>© 2024 DataAiPrep</p>
        </footer>
    </div>
</body>
</html>
"""
    
    def save(
        self,
        filepath: str,
        format: str = 'html'
    ) -> str:
        """
        Save report to file.
        
        Args:
            filepath: Output file path
            format: Output format ('html', 'pdf', 'json')
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        
        if format == 'html':
            content = self.generate_html()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif format == 'json':
            content = {
                'title': self.title,
                'generated_at': datetime.now().isoformat(),
                'sections': [
                    {
                        'title': s.title,
                        'type': s.content_type,
                        'content': s.content
                    }
                    for s in self._sections
                ]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, default=str)
                
        elif format == 'pdf':
            # Try to generate PDF using weasyprint or pdfkit
            html_content = self.generate_html()
            
            try:
                import weasyprint
                pdf = weasyprint.HTML(string=html_content).write_pdf()
                with open(filepath, 'wb') as f:
                    f.write(pdf)
            except ImportError:
                try:
                    import pdfkit
                    pdfkit.from_string(html_content, str(filepath))
                except ImportError:
                    # Fallback: save as HTML with PDF extension note
                    html_path = filepath.with_suffix('.html')
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    print(f"PDF generation requires weasyprint or pdfkit. Saved as HTML: {html_path}")
                    return str(html_path)
        
        return str(filepath)


class ModelCardGenerator:
    """
    Generate Model Cards for ML model documentation.
    
    Based on Google's Model Cards framework for responsible AI documentation.
    """
    
    def __init__(self):
        """Initialize the model card generator"""
        self.card_data = {
            'model_details': {},
            'intended_use': {},
            'factors': {},
            'metrics': {},
            'evaluation_data': {},
            'training_data': {},
            'ethical_considerations': {},
            'caveats_recommendations': {}
        }
    
    def set_model_details(
        self,
        name: str,
        version: str = "1.0",
        type: str = "",
        description: str = "",
        developers: List[str] = None,
        license: str = ""
    ):
        """Set model details section"""
        self.card_data['model_details'] = {
            'name': name,
            'version': version,
            'type': type,
            'description': description,
            'developers': developers or [],
            'license': license,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def set_intended_use(
        self,
        primary_use: str,
        users: List[str] = None,
        out_of_scope: List[str] = None
    ):
        """Set intended use section"""
        self.card_data['intended_use'] = {
            'primary_use': primary_use,
            'intended_users': users or [],
            'out_of_scope_uses': out_of_scope or []
        }
    
    def set_metrics(
        self,
        metrics: Dict[str, float],
        thresholds: Dict[str, float] = None
    ):
        """Set model metrics"""
        self.card_data['metrics'] = {
            'values': metrics,
            'thresholds': thresholds or {}
        }
    
    def set_ethical_considerations(
        self,
        risks: List[str] = None,
        mitigations: List[str] = None,
        bias_analysis: str = ""
    ):
        """Set ethical considerations"""
        self.card_data['ethical_considerations'] = {
            'risks': risks or [],
            'mitigations': mitigations or [],
            'bias_analysis': bias_analysis
        }
    
    def generate_markdown(self) -> str:
        """Generate model card as Markdown"""
        md = []
        details = self.card_data['model_details']
        
        md.append(f"# Model Card: {details.get('name', 'Unnamed Model')}")
        md.append("")
        
        # Model Details
        md.append("## Model Details")
        md.append(f"- **Version:** {details.get('version', 'N/A')}")
        md.append(f"- **Type:** {details.get('type', 'N/A')}")
        md.append(f"- **Date:** {details.get('date', 'N/A')}")
        if details.get('developers'):
            md.append(f"- **Developers:** {', '.join(details['developers'])}")
        md.append("")
        md.append(f"### Description")
        md.append(details.get('description', 'No description provided.'))
        md.append("")
        
        # Intended Use
        intended = self.card_data['intended_use']
        if intended:
            md.append("## Intended Use")
            md.append(f"**Primary Use:** {intended.get('primary_use', 'N/A')}")
            md.append("")
            if intended.get('intended_users'):
                md.append("**Intended Users:**")
                for user in intended['intended_users']:
                    md.append(f"- {user}")
            if intended.get('out_of_scope_uses'):
                md.append("")
                md.append("**Out of Scope:**")
                for use in intended['out_of_scope_uses']:
                    md.append(f"- {use}")
            md.append("")
        
        # Metrics
        metrics = self.card_data['metrics']
        if metrics.get('values'):
            md.append("## Performance Metrics")
            md.append("")
            md.append("| Metric | Value | Threshold |")
            md.append("|--------|-------|-----------|")
            for metric, value in metrics['values'].items():
                threshold = metrics.get('thresholds', {}).get(metric, '-')
                md.append(f"| {metric} | {value:.4f} | {threshold} |")
            md.append("")
        
        # Ethical Considerations
        ethical = self.card_data['ethical_considerations']
        if any(ethical.values()):
            md.append("## Ethical Considerations")
            md.append("")
            if ethical.get('risks'):
                md.append("### Risks")
                for risk in ethical['risks']:
                    md.append(f"- {risk}")
                md.append("")
            if ethical.get('mitigations'):
                md.append("### Mitigations")
                for mitigation in ethical['mitigations']:
                    md.append(f"- {mitigation}")
                md.append("")
            if ethical.get('bias_analysis'):
                md.append("### Bias Analysis")
                md.append(ethical['bias_analysis'])
                md.append("")
        
        md.append("---")
        md.append(f"*Generated by DataAiPrep on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return '\n'.join(md)
    
    def save(self, filepath: str):
        """Save model card to file"""
        content = self.generate_markdown()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

