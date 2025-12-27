"""
Results tabs widget for displaying analysis results
"""

from PyQt6.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QTableWidget, QTableWidgetItem, QScrollArea,
    QGroupBox, QGridLayout, QPushButton, QSplitter, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
import matplotlib
matplotlib.use('Qt5Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd


class MatplotlibWidget(QWidget):
    """Widget for embedding matplotlib plots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        
    def plot(self, plot_func, *args, **kwargs):
        """Execute a plotting function and refresh the canvas"""
        self.figure.clear()
        plot_func(self.figure, *args, **kwargs)
        self.canvas.draw()


class OverviewTab(QWidget):
    """Overview tab showing dataset summary"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Dataset info section
        info_group = QGroupBox("Dataset Information")
        info_layout = QGridLayout(info_group)
        
        self.shape_label = QLabel("Shape: -")
        self.memory_label = QLabel("Memory Usage: -")
        self.dtypes_label = QLabel("Data Types: -")
        
        info_layout.addWidget(QLabel("Dimensions:"), 0, 0)
        info_layout.addWidget(self.shape_label, 0, 1)
        info_layout.addWidget(QLabel("Memory:"), 1, 0)
        info_layout.addWidget(self.memory_label, 1, 1)
        info_layout.addWidget(QLabel("Types:"), 2, 0)
        info_layout.addWidget(self.dtypes_label, 2, 1)
        
        layout.addWidget(info_group)
        
        # Quality score section
        score_group = QGroupBox("Data Quality Score")
        score_layout = QVBoxLayout(score_group)
        
        self.quality_score_label = QLabel("Quality Score: Calculating...")
        score_font = QFont()
        score_font.setPointSize(16)
        score_font.setBold(True)
        self.quality_score_label.setFont(score_font)
        self.quality_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(self.quality_score_label)
        
        self.score_breakdown = QTextEdit()
        self.score_breakdown.setMaximumHeight(100)
        score_layout.addWidget(self.score_breakdown)
        
        layout.addWidget(score_group)
        
        # Column information table
        columns_group = QGroupBox("Column Information")
        columns_layout = QVBoxLayout(columns_group)
        
        self.columns_table = QTableWidget()
        self.columns_table.setColumnCount(4)
        self.columns_table.setHorizontalHeaderLabels(["Column", "Type", "Non-Null Count", "Missing %"])
        columns_layout.addWidget(self.columns_table)
        
        layout.addWidget(columns_group)
        
        layout.addStretch()
        
    def update_data(self, data_info: dict, completeness_info: dict = None):
        """Update the overview with data information"""
        # Update basic info
        shape = data_info.get('shape', [0, 0])
        self.shape_label.setText(f"{shape[0]:,} rows × {shape[1]} columns")
        
        memory_mb = data_info.get('memory_usage', 0) / 1024**2
        self.memory_label.setText(f"{memory_mb:.2f} MB")
        
        dtypes = data_info.get('dtypes', {})
        dtype_counts = {}
        for dtype in dtypes.values():
            dtype_str = str(dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        
        dtype_summary = ", ".join([f"{count} {dtype}" for dtype, count in dtype_counts.items()])
        self.dtypes_label.setText(dtype_summary)
        
        # Update quality score (simplified calculation for MVP)
        if completeness_info:
            missing_pct = completeness_info.get('overall_missing_percentage', 0)
            quality_score = max(0, 100 - missing_pct * 2)  # Simple calculation
            
            self.quality_score_label.setText(f"Quality Score: {quality_score:.0f}/100")
            
            # Color code the score
            if quality_score >= 80:
                color = "#4CAF50"  # Green
            elif quality_score >= 60:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
                
            self.quality_score_label.setStyleSheet(f"color: {color};")
            
            breakdown_text = f"""Completeness: {100 - missing_pct:.1f}%
Distribution: Analyzing...
Consistency: Analyzing...
Relevance: Analyzing..."""
            self.score_breakdown.setPlainText(breakdown_text)
        
        # Update columns table
        columns = data_info.get('columns', [])
        self.columns_table.setRowCount(len(columns))
        
        missing_by_column = {}
        if completeness_info:
            missing_by_column = completeness_info.get('missing_by_column', {})
        
        for i, column in enumerate(columns):
            self.columns_table.setItem(i, 0, QTableWidgetItem(column))
            self.columns_table.setItem(i, 1, QTableWidgetItem(str(dtypes.get(column, 'unknown'))))
            
            total_rows = shape[0]
            missing_count = missing_by_column.get(column, 0)
            non_null_count = total_rows - missing_count
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
            
            self.columns_table.setItem(i, 2, QTableWidgetItem(f"{non_null_count:,}"))
            self.columns_table.setItem(i, 3, QTableWidgetItem(f"{missing_pct:.1f}%"))
        
        # Resize columns to content
        self.columns_table.resizeColumnsToContents()


class CompletenessTab(QWidget):
    """Tab for data completeness analysis results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Summary section
        summary_group = QGroupBox("Completeness Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.overall_missing_label = QLabel("Overall Missing: -")
        self.complete_cols_label = QLabel("Complete Columns: -")
        self.incomplete_cols_label = QLabel("Incomplete Columns: -")
        self.most_missing_label = QLabel("Most Missing Column: -")
        
        summary_layout.addWidget(QLabel("Overall Missing:"), 0, 0)
        summary_layout.addWidget(self.overall_missing_label, 0, 1)
        summary_layout.addWidget(QLabel("Complete Columns:"), 1, 0)
        summary_layout.addWidget(self.complete_cols_label, 1, 1)
        summary_layout.addWidget(QLabel("Incomplete Columns:"), 2, 0)
        summary_layout.addWidget(self.incomplete_cols_label, 2, 1)
        summary_layout.addWidget(QLabel("Most Missing:"), 3, 0)
        summary_layout.addWidget(self.most_missing_label, 3, 1)
        
        layout.addWidget(summary_group)
        
        # Create splitter for plots and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Missing data visualization
        plot_widget = MatplotlibWidget()
        self.plot_widget = plot_widget
        splitter.addWidget(plot_widget)
        
        # Missing data details
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        details_group = QGroupBox("Missing Data Details")
        details_group_layout = QVBoxLayout(details_group)
        
        self.missing_table = QTableWidget()
        self.missing_table.setColumnCount(3)
        self.missing_table.setHorizontalHeaderLabels(["Column", "Missing Count", "Missing %"])
        details_group_layout.addWidget(self.missing_table)
        
        details_layout.addWidget(details_group)
        
        # Recommendations
        rec_group = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout(rec_group)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setMaximumHeight(150)
        rec_layout.addWidget(self.recommendations_text)
        
        details_layout.addWidget(rec_group)
        
        splitter.addWidget(details_widget)
        
        # Set splitter proportions
        splitter.setSizes([600, 400])
        
    def update_data(self, completeness_info: dict):
        """Update the completeness tab with analysis results"""
        # Update summary
        overall_missing = completeness_info.get('overall_missing_percentage', 0)
        self.overall_missing_label.setText(f"{overall_missing:.2f}%")
        
        complete_count = completeness_info.get('complete_columns_count', 0)
        self.complete_cols_label.setText(str(complete_count))
        
        incomplete_count = completeness_info.get('incomplete_columns_count', 0)
        self.incomplete_cols_label.setText(str(incomplete_count))
        
        most_missing = completeness_info.get('most_missing_column', {})
        if most_missing:
            most_missing_text = f"{most_missing.get('column', 'N/A')} ({most_missing.get('percentage', 0):.1f}%)"
        else:
            most_missing_text = "None"
        self.most_missing_label.setText(most_missing_text)
        
        # Update missing data table
        missing_by_column = completeness_info.get('missing_by_column', {})
        self.missing_table.setRowCount(len(missing_by_column))
        
        total_rows = completeness_info.get('total_rows', 1)
        
        for i, (column, missing_count) in enumerate(missing_by_column.items()):
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
            
            self.missing_table.setItem(i, 0, QTableWidgetItem(column))
            self.missing_table.setItem(i, 1, QTableWidgetItem(str(missing_count)))
            self.missing_table.setItem(i, 2, QTableWidgetItem(f"{missing_pct:.2f}%"))
        
        # Sort by missing percentage
        self.missing_table.sortItems(2, Qt.SortOrder.DescendingOrder)
        self.missing_table.resizeColumnsToContents()
        
        # Create visualization
        self.create_missing_plot(missing_by_column, total_rows)
        
        # Update recommendations
        recommendations = self.generate_recommendations(completeness_info)
        self.recommendations_text.setPlainText(recommendations)
        
    def create_missing_plot(self, missing_by_column: dict, total_rows: int):
        """Create missing data visualization"""
        def plot_missing_data(figure, missing_data, total_rows):
            if not missing_data:
                return
                
            # Prepare data
            columns = list(missing_data.keys())
            missing_counts = list(missing_data.values())
            missing_percentages = [(count / total_rows * 100) if total_rows > 0 else 0 
                                 for count in missing_counts]
            
            # Create subplot
            ax = figure.add_subplot(111)
            
            # Only show columns with missing data
            columns_with_missing = [(col, pct) for col, pct in zip(columns, missing_percentages) if pct > 0]
            
            if columns_with_missing:
                cols, pcts = zip(*columns_with_missing)
                
                # Horizontal bar chart
                y_pos = range(len(cols))
                bars = ax.barh(y_pos, pcts, color='lightcoral')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(cols)
                ax.set_xlabel('Missing Percentage (%)')
                ax.set_title('Missing Data by Column')
                
                # Add percentage labels on bars
                for i, (bar, pct) in enumerate(zip(bars, pcts)):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                           f'{pct:.1f}%', ha='left', va='center')
                
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No missing data found!', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=16, color='green')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title('Missing Data Analysis')
            
            figure.tight_layout()
        
        self.plot_widget.plot(plot_missing_data, missing_by_column, total_rows)
        
    def generate_recommendations(self, completeness_info: dict) -> str:
        """Generate recommendations based on completeness analysis"""
        recommendations = []
        
        overall_missing = completeness_info.get('overall_missing_percentage', 0)
        incomplete_count = completeness_info.get('incomplete_columns_count', 0)
        
        if overall_missing < 5:
            recommendations.append("✓ Excellent data completeness! Dataset is ready for ML training.")
        elif overall_missing < 15:
            recommendations.append("⚠ Good data completeness with minor missing values.")
            recommendations.append("• Consider simple imputation strategies (mean, median, mode)")
        else:
            recommendations.append("⚠ Significant missing data detected!")
            recommendations.append("• Investigate missing data patterns (MCAR vs MAR vs MNAR)")
            recommendations.append("• Consider advanced imputation methods (KNN, iterative)")
            recommendations.append("• Evaluate if data collection process can be improved")
        
        if incomplete_count > 0:
            recommendations.append(f"• {incomplete_count} columns have missing values")
            recommendations.append("• Review each column's business importance")
            recommendations.append("• Consider dropping columns with >50% missing data")
        
        return "\n".join(recommendations)


class DistributionTab(QWidget):
    """Tab for data distribution analysis results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Summary section
        summary_group = QGroupBox("Distribution Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.numeric_cols_label = QLabel("Numeric Columns: -")
        self.categorical_cols_label = QLabel("Categorical Columns: -")
        self.outliers_label = QLabel("Outliers Detected: -")
        self.skewed_cols_label = QLabel("Highly Skewed: -")
        
        summary_layout.addWidget(QLabel("Numeric Columns:"), 0, 0)
        summary_layout.addWidget(self.numeric_cols_label, 0, 1)
        summary_layout.addWidget(QLabel("Categorical Columns:"), 1, 0)
        summary_layout.addWidget(self.categorical_cols_label, 1, 1)
        summary_layout.addWidget(QLabel("Outliers:"), 2, 0)
        summary_layout.addWidget(self.outliers_label, 2, 1)
        summary_layout.addWidget(QLabel("Skewed Columns:"), 3, 0)
        summary_layout.addWidget(self.skewed_cols_label, 3, 1)
        
        layout.addWidget(summary_group)
        
        # Tabbed results
        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)
        
        # Statistical summary tab
        self.stats_tab = self.create_stats_tab()
        self.sub_tabs.addTab(self.stats_tab, "Statistical Summary")
        
        # Distributions plot tab
        self.dist_plot_tab = MatplotlibWidget()
        self.sub_tabs.addTab(self.dist_plot_tab, "Distributions")
        
        # Outliers tab
        self.outliers_tab = self.create_outliers_tab()
        self.sub_tabs.addTab(self.outliers_tab, "Outliers")
        
    def create_stats_tab(self) -> QWidget:
        """Create statistical summary tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.stats_table = QTableWidget()
        layout.addWidget(self.stats_table)
        
        return widget
        
    def create_outliers_tab(self) -> QWidget:
        """Create outliers analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.outliers_table = QTableWidget()
        self.outliers_table.setColumnCount(4)
        self.outliers_table.setHorizontalHeaderLabels(["Column", "Method", "Outliers Count", "Percentage"])
        layout.addWidget(self.outliers_table)
        
        return widget
        
    def update_data(self, distribution_info: dict):
        """Update the distribution tab with analysis results"""
        # Update summary
        numeric_count = len(distribution_info.get('numeric_columns', []))
        categorical_count = len(distribution_info.get('categorical_columns', []))
        
        self.numeric_cols_label.setText(str(numeric_count))
        self.categorical_cols_label.setText(str(categorical_count))
        
        # Outliers summary
        outliers_info = distribution_info.get('outliers', {})
        total_outliers = 0
        if outliers_info:
            for col_outliers in outliers_info.values():
                if isinstance(col_outliers, dict) and 'iqr' in col_outliers:
                    total_outliers += col_outliers['iqr'].get('count', 0)
        self.outliers_label.setText(str(total_outliers))
        
        # Skewed columns
        skewed_cols = distribution_info.get('highly_skewed_columns', [])
        self.skewed_cols_label.setText(str(len(skewed_cols)))
        
        # Update statistical summary
        self.update_stats_table(distribution_info.get('statistical_summary', {}))
        
        # Update outliers table
        if outliers_info:
            self.update_outliers_table(outliers_info)
        
        # Create distribution plots
        self.create_distribution_plots(distribution_info)
        
    def update_stats_table(self, stats_summary: dict):
        """Update statistical summary table"""
        if not stats_summary:
            return
            
        columns = list(stats_summary.keys())
        stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] if columns else []
        
        self.stats_table.setRowCount(len(stats))
        self.stats_table.setColumnCount(len(columns) + 1)
        
        headers = ['Statistic'] + columns
        self.stats_table.setHorizontalHeaderLabels(headers)
        
        for i, stat in enumerate(stats):
            self.stats_table.setItem(i, 0, QTableWidgetItem(stat))
            for j, col in enumerate(columns):
                value = stats_summary[col].get(stat, '')
                if isinstance(value, float):
                    value = f"{value:.4f}"
                self.stats_table.setItem(i, j + 1, QTableWidgetItem(str(value)))
        
        self.stats_table.resizeColumnsToContents()
        
    def update_outliers_table(self, outliers_info: dict):
        """Update outliers table"""
        self.outliers_table.setRowCount(len(outliers_info))
        
        for i, (column, outlier_data) in enumerate(outliers_info.items()):
            self.outliers_table.setItem(i, 0, QTableWidgetItem(column))
            self.outliers_table.setItem(i, 1, QTableWidgetItem("IQR Method"))
            
            # Handle different data structures
            if isinstance(outlier_data, dict) and 'iqr' in outlier_data:
                iqr_count = outlier_data['iqr'].get('count', 0)
                iqr_percentage = outlier_data['iqr'].get('percentage', 0)
                self.outliers_table.setItem(i, 2, QTableWidgetItem(str(iqr_count)))
                self.outliers_table.setItem(i, 3, QTableWidgetItem(f"{iqr_percentage:.1f}%"))
            else:
                # Fallback for unexpected data structure
                self.outliers_table.setItem(i, 2, QTableWidgetItem(str(outlier_data)))
                self.outliers_table.setItem(i, 3, QTableWidgetItem("N/A"))
        
        self.outliers_table.resizeColumnsToContents()
        
    def create_distribution_plots(self, distribution_info: dict):
        """Create distribution visualization plots"""
        def plot_distributions(figure, dist_info):
            numeric_cols = dist_info.get('numeric_columns', [])
            
            if not numeric_cols:
                ax = figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No numeric columns to plot distributions', 
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # For now, just create a placeholder
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Distribution plots for {len(numeric_cols)} numeric columns\n(Implementation in progress)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Data Distributions')
            
            figure.tight_layout()
        
        self.dist_plot_tab.plot(plot_distributions, distribution_info)


class DataLeakageTab(QWidget):
    """Tab for data leakage detection results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Summary section
        summary_group = QGroupBox("Leakage Detection Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.leakage_score_label = QLabel("Leakage Risk Score: -")
        self.perfect_corr_label = QLabel("Perfect Correlations: -")
        self.duplicate_features_label = QLabel("Duplicate Features: -")
        self.target_leakage_label = QLabel("Target Leakage Issues: -")
        
        summary_layout.addWidget(QLabel("Risk Score:"), 0, 0)
        summary_layout.addWidget(self.leakage_score_label, 0, 1)
        summary_layout.addWidget(QLabel("Perfect Correlations:"), 1, 0)
        summary_layout.addWidget(self.perfect_corr_label, 1, 1)
        summary_layout.addWidget(QLabel("Duplicate Features:"), 2, 0)
        summary_layout.addWidget(self.duplicate_features_label, 2, 1)
        summary_layout.addWidget(QLabel("Target Leakage:"), 3, 0)
        summary_layout.addWidget(self.target_leakage_label, 3, 1)
        
        layout.addWidget(summary_group)
        
        # Tabbed detailed results
        self.leakage_sub_tabs = QTabWidget()
        layout.addWidget(self.leakage_sub_tabs)
        
        # Perfect correlations tab
        self.perfect_corr_tab = self.create_perfect_corr_tab()
        self.leakage_sub_tabs.addTab(self.perfect_corr_tab, "Perfect Correlations")
        
        # Target leakage tab
        self.target_leakage_tab = self.create_target_leakage_tab()
        self.leakage_sub_tabs.addTab(self.target_leakage_tab, "Target Leakage")
        
        # Recommendations tab
        self.leakage_recommendations_tab = self.create_recommendations_tab()
        self.leakage_sub_tabs.addTab(self.leakage_recommendations_tab, "Recommendations")
        
    def create_perfect_corr_tab(self) -> QWidget:
        """Create perfect correlations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.perfect_corr_table = QTableWidget()
        self.perfect_corr_table.setColumnCount(5)
        self.perfect_corr_table.setHorizontalHeaderLabels(
            ["Feature 1", "Feature 2", "Correlation", "Type", "Severity"]
        )
        layout.addWidget(self.perfect_corr_table)
        
        return widget
        
    def create_target_leakage_tab(self) -> QWidget:
        """Create target leakage tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.target_leakage_table = QTableWidget()
        self.target_leakage_table.setColumnCount(4)
        self.target_leakage_table.setHorizontalHeaderLabels(
            ["Feature", "Issue Type", "Score/Details", "Severity"]
        )
        layout.addWidget(self.target_leakage_table)
        
        return widget
        
    def create_recommendations_tab(self) -> QWidget:
        """Create recommendations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.leakage_recommendations_text = QTextEdit()
        self.leakage_recommendations_text.setReadOnly(True)
        layout.addWidget(self.leakage_recommendations_text)
        
        return widget
        
    def update_data(self, leakage_info: dict):
        """Update the leakage tab with analysis results"""
        # Update summary
        leakage_score = leakage_info.get('leakage_score', 0)
        self.leakage_score_label.setText(f"{leakage_score:.1f}/100")
        
        # Color code the score
        if leakage_score <= 20:
            color = "#4CAF50"  # Green (low risk)
        elif leakage_score <= 50:
            color = "#FF9800"  # Orange (medium risk)
        else:
            color = "#F44336"  # Red (high risk)
        self.leakage_score_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        perfect_corrs = len(leakage_info.get('perfect_correlations', []))
        self.perfect_corr_label.setText(str(perfect_corrs))
        
        duplicates = len(leakage_info.get('duplicate_features', []))
        self.duplicate_features_label.setText(str(duplicates))
        
        target_leakage = leakage_info.get('target_leakage', {})
        target_issues = (len(target_leakage.get('perfect_predictors', [])) + 
                        len(target_leakage.get('high_information_features', [])))
        self.target_leakage_label.setText(str(target_issues))
        
        # Update perfect correlations table
        self.update_perfect_corr_table(leakage_info.get('perfect_correlations', []))
        
        # Update target leakage table
        self.update_target_leakage_table(leakage_info.get('target_leakage', {}))
        
        # Update recommendations
        recommendations = leakage_info.get('recommendations', [])
        self.leakage_recommendations_text.setPlainText('\n'.join(recommendations))
        
    def update_perfect_corr_table(self, perfect_correlations: list):
        """Update perfect correlations table"""
        self.perfect_corr_table.setRowCount(len(perfect_correlations))
        
        for i, corr in enumerate(perfect_correlations):
            self.perfect_corr_table.setItem(i, 0, QTableWidgetItem(corr['feature1']))
            self.perfect_corr_table.setItem(i, 1, QTableWidgetItem(corr['feature2']))
            self.perfect_corr_table.setItem(i, 2, QTableWidgetItem(f"{corr['correlation']:.4f}"))
            self.perfect_corr_table.setItem(i, 3, QTableWidgetItem(corr['type']))
            self.perfect_corr_table.setItem(i, 4, QTableWidgetItem(corr['severity']))
        
        self.perfect_corr_table.resizeColumnsToContents()
        
    def update_target_leakage_table(self, target_leakage: dict):
        """Update target leakage table"""
        leakage_items = []
        
        # Perfect predictors
        for pred in target_leakage.get('perfect_predictors', []):
            leakage_items.append({
                'feature': pred['feature'],
                'type': 'Perfect Predictor',
                'details': 'Perfect prediction',
                'severity': pred['severity']
            })
            
        # High information features
        for feat in target_leakage.get('high_information_features', []):
            leakage_items.append({
                'feature': feat['feature'],
                'type': 'High Information',
                'details': f"Score: {feat['information_score']:.4f}",
                'severity': feat['severity']
            })
            
        # Suspicious names
        for name in target_leakage.get('suspicious_names', []):
            leakage_items.append({
                'feature': name['feature'],
                'type': 'Suspicious Name',
                'details': name['reason'],
                'severity': name['severity']
            })
        
        self.target_leakage_table.setRowCount(len(leakage_items))
        
        for i, item in enumerate(leakage_items):
            self.target_leakage_table.setItem(i, 0, QTableWidgetItem(item['feature']))
            self.target_leakage_table.setItem(i, 1, QTableWidgetItem(item['type']))
            self.target_leakage_table.setItem(i, 2, QTableWidgetItem(item['details']))
            self.target_leakage_table.setItem(i, 3, QTableWidgetItem(item['severity']))
        
        self.target_leakage_table.resizeColumnsToContents()


class FeatureQualityTab(QWidget):
    """Tab for feature quality assessment results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Summary section
        summary_group = QGroupBox("Feature Quality Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.quality_score_label = QLabel("Quality Score: -")
        self.low_variance_label = QLabel("Low Variance Features: -")
        self.redundant_label = QLabel("Redundant Features: -")
        self.irrelevant_label = QLabel("Irrelevant Features: -")
        
        summary_layout.addWidget(QLabel("Overall Score:"), 0, 0)
        summary_layout.addWidget(self.quality_score_label, 0, 1)
        summary_layout.addWidget(QLabel("Low Variance:"), 1, 0)
        summary_layout.addWidget(self.low_variance_label, 1, 1)
        summary_layout.addWidget(QLabel("Redundant:"), 2, 0)
        summary_layout.addWidget(self.redundant_label, 2, 1)
        summary_layout.addWidget(QLabel("Irrelevant:"), 3, 0)
        summary_layout.addWidget(self.irrelevant_label, 3, 1)
        
        layout.addWidget(summary_group)
        
        # Tabbed detailed results
        self.quality_sub_tabs = QTabWidget()
        layout.addWidget(self.quality_sub_tabs)
        
        # Feature scores tab
        self.feature_scores_tab = self.create_feature_scores_tab()
        self.quality_sub_tabs.addTab(self.feature_scores_tab, "Feature Scores")
        
        # Feature importance tab
        self.importance_tab = self.create_importance_tab()
        self.quality_sub_tabs.addTab(self.importance_tab, "Feature Importance")
        
        # Issues tab
        self.quality_issues_tab = self.create_quality_issues_tab()
        self.quality_sub_tabs.addTab(self.quality_issues_tab, "Quality Issues")
        
        # Recommendations tab
        self.quality_recommendations_tab = self.create_quality_recommendations_tab()
        self.quality_sub_tabs.addTab(self.quality_recommendations_tab, "Recommendations")
        
    def create_feature_scores_tab(self) -> QWidget:
        """Create feature scores tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.feature_scores_table = QTableWidget()
        self.feature_scores_table.setColumnCount(3)
        self.feature_scores_table.setHorizontalHeaderLabels(
            ["Feature", "Quality Score", "Grade"]
        )
        layout.addWidget(self.feature_scores_table)
        
        return widget
        
    def create_importance_tab(self) -> QWidget:
        """Create feature importance tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(3)
        self.importance_table.setHorizontalHeaderLabels(
            ["Feature", "Importance Score", "Rank"]
        )
        layout.addWidget(self.importance_table)
        
        return widget
        
    def create_quality_issues_tab(self) -> QWidget:
        """Create quality issues tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.quality_issues_table = QTableWidget()
        self.quality_issues_table.setColumnCount(4)
        self.quality_issues_table.setHorizontalHeaderLabels(
            ["Feature", "Issue Type", "Details", "Recommendation"]
        )
        layout.addWidget(self.quality_issues_table)
        
        return widget
        
    def create_quality_recommendations_tab(self) -> QWidget:
        """Create quality recommendations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.quality_recommendations_text = QTextEdit()
        self.quality_recommendations_text.setReadOnly(True)
        layout.addWidget(self.quality_recommendations_text)
        
        return widget
        
    def update_data(self, quality_info: dict):
        """Update the feature quality tab with analysis results"""
        # Update summary
        quality_score = quality_info.get('quality_score', 0)
        self.quality_score_label.setText(f"{quality_score:.1f}/100")
        
        # Color code the score
        if quality_score >= 80:
            color = "#4CAF50"  # Green
        elif quality_score >= 60:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        self.quality_score_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        low_variance = len(quality_info.get('low_variance_features', []))
        self.low_variance_label.setText(str(low_variance))
        
        redundant = len(quality_info.get('redundant_features', []))
        self.redundant_label.setText(str(redundant))
        
        irrelevant = len(quality_info.get('irrelevant_features', []))
        self.irrelevant_label.setText(str(irrelevant))
        
        # Update feature scores table
        self.update_feature_scores_table(quality_info.get('feature_quality_scores', {}))
        
        # Update importance table
        self.update_importance_table(quality_info.get('feature_importance', {}))
        
        # Update quality issues table
        self.update_quality_issues_table(quality_info)
        
        # Update recommendations
        recommendations = quality_info.get('recommendations', [])
        self.quality_recommendations_text.setPlainText('\n'.join(recommendations))
        
    def update_feature_scores_table(self, feature_scores: dict):
        """Update feature scores table"""
        # Sort features by score (descending)
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        self.feature_scores_table.setRowCount(len(sorted_features))
        
        for i, (feature, score) in enumerate(sorted_features):
            self.feature_scores_table.setItem(i, 0, QTableWidgetItem(feature))
            self.feature_scores_table.setItem(i, 1, QTableWidgetItem(f"{score:.2f}"))
            
            # Assign grade
            if score >= 90:
                grade = "A"
            elif score >= 80:
                grade = "B"
            elif score >= 70:
                grade = "C"
            elif score >= 60:
                grade = "D"
            else:
                grade = "F"
            self.feature_scores_table.setItem(i, 2, QTableWidgetItem(grade))
        
        self.feature_scores_table.resizeColumnsToContents()
        
    def update_importance_table(self, feature_importance: dict):
        """Update feature importance table"""
        # Sort features by importance (descending)
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.importance_table.setRowCount(len(sorted_importance))
        
        for i, (feature, importance) in enumerate(sorted_importance):
            self.importance_table.setItem(i, 0, QTableWidgetItem(feature))
            self.importance_table.setItem(i, 1, QTableWidgetItem(f"{importance:.6f}"))
            self.importance_table.setItem(i, 2, QTableWidgetItem(str(i + 1)))
        
        self.importance_table.resizeColumnsToContents()
        
    def update_quality_issues_table(self, quality_info: dict):
        """Update quality issues table"""
        issues = []
        
        # Low variance features
        for feature in quality_info.get('low_variance_features', []):
            issues.append({
                'feature': feature['feature'],
                'type': 'Low Variance',
                'details': f"Variance: {feature['variance']:.6f}",
                'recommendation': 'Consider removing'
            })
            
        # Redundant features
        for feature in quality_info.get('redundant_features', []):
            issues.append({
                'feature': feature['feature2'],
                'type': 'Redundant',
                'details': f"Corr with {feature['feature1']}: {feature['correlation']:.3f}",
                'recommendation': feature['recommendation']
            })
            
        # High cardinality features
        for feature in quality_info.get('high_cardinality_features', []):
            issues.append({
                'feature': feature['feature'],
                'type': 'High Cardinality',
                'details': f"{feature['unique_count']} unique values",
                'recommendation': feature['recommendation']
            })
            
        # Irrelevant features
        for feature in quality_info.get('irrelevant_features', []):
            issues.append({
                'feature': feature['feature'],
                'type': 'Low Importance',
                'details': f"Score: {feature['importance_score']:.6f}",
                'recommendation': 'Consider removing'
            })
        
        self.quality_issues_table.setRowCount(len(issues))
        
        for i, issue in enumerate(issues):
            self.quality_issues_table.setItem(i, 0, QTableWidgetItem(issue['feature']))
            self.quality_issues_table.setItem(i, 1, QTableWidgetItem(issue['type']))
            self.quality_issues_table.setItem(i, 2, QTableWidgetItem(issue['details']))
            self.quality_issues_table.setItem(i, 3, QTableWidgetItem(issue['recommendation']))
        
        self.quality_issues_table.resizeColumnsToContents()


class DimensionalityTab(QWidget):
    """Tab for PCA and t-SNE dimensionality reduction results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Summary section
        summary_group = QGroupBox("Dimensionality Reduction Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.reduction_score_label = QLabel("Reduction Score: -")
        self.total_features_label = QLabel("Total Features: -")
        self.pca_components_label = QLabel("PCA Components (90%): -")
        self.tsne_quality_label = QLabel("t-SNE Quality: -")
        
        summary_layout.addWidget(QLabel("Overall Score:"), 0, 0)
        summary_layout.addWidget(self.reduction_score_label, 0, 1)
        summary_layout.addWidget(QLabel("Features:"), 1, 0)
        summary_layout.addWidget(self.total_features_label, 1, 1)
        summary_layout.addWidget(QLabel("PCA 90% Comp:"), 2, 0)
        summary_layout.addWidget(self.pca_components_label, 2, 1)
        summary_layout.addWidget(QLabel("t-SNE Quality:"), 3, 0)
        summary_layout.addWidget(self.tsne_quality_label, 3, 1)
        
        layout.addWidget(summary_group)
        
        # Tabbed results
        self.dim_sub_tabs = QTabWidget()
        layout.addWidget(self.dim_sub_tabs)
        
        # PCA Analysis tab
        self.pca_tab = self.create_pca_tab()
        self.dim_sub_tabs.addTab(self.pca_tab, "PCA Analysis")
        
        # t-SNE Analysis tab
        self.tsne_tab = self.create_tsne_tab()
        self.dim_sub_tabs.addTab(self.tsne_tab, "t-SNE Analysis")
        
        # Feature Importance tab
        self.importance_tab = self.create_feature_importance_tab()
        self.dim_sub_tabs.addTab(self.importance_tab, "Feature Importance")
        
        # Visualizations tab
        self.visualizations_tab = self.create_visualizations_tab()
        self.dim_sub_tabs.addTab(self.visualizations_tab, "Visualizations")
        
        # Recommendations tab
        self.dim_recommendations_tab = self.create_dim_recommendations_tab()
        self.dim_sub_tabs.addTab(self.dim_recommendations_tab, "Recommendations")
        
    def create_pca_tab(self) -> QWidget:
        """Create PCA analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # PCA variance explanation table
        variance_group = QGroupBox("Explained Variance by Components")
        variance_layout = QVBoxLayout(variance_group)
        
        self.pca_variance_table = QTableWidget()
        self.pca_variance_table.setColumnCount(4)
        self.pca_variance_table.setHorizontalHeaderLabels(
            ["Component", "Variance Explained", "Cumulative Variance", "Percentage"]
        )
        variance_layout.addWidget(self.pca_variance_table)
        layout.addWidget(variance_group)
        
        # PCA summary statistics
        stats_group = QGroupBox("PCA Summary Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.pca_components_90_label = QLabel("-")
        self.pca_components_95_label = QLabel("-")
        self.pca_total_variance_label = QLabel("-")
        
        stats_layout.addWidget(QLabel("Components for 90% variance:"), 0, 0)
        stats_layout.addWidget(self.pca_components_90_label, 0, 1)
        stats_layout.addWidget(QLabel("Components for 95% variance:"), 1, 0)
        stats_layout.addWidget(self.pca_components_95_label, 1, 1)
        stats_layout.addWidget(QLabel("Total variance captured:"), 2, 0)
        stats_layout.addWidget(self.pca_total_variance_label, 2, 1)
        
        layout.addWidget(stats_group)
        
        return widget
        
    def create_tsne_tab(self) -> QWidget:
        """Create t-SNE analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # t-SNE parameters and results
        tsne_group = QGroupBox("t-SNE Analysis Results")
        tsne_layout = QGridLayout(tsne_group)
        
        self.tsne_samples_label = QLabel("-")
        self.tsne_features_label = QLabel("-")
        self.tsne_perplexity_label = QLabel("-")
        self.tsne_kl_divergence_label = QLabel("-")
        self.tsne_preprocessing_label = QLabel("-")
        
        tsne_layout.addWidget(QLabel("Samples analyzed:"), 0, 0)
        tsne_layout.addWidget(self.tsne_samples_label, 0, 1)
        tsne_layout.addWidget(QLabel("Features used:"), 1, 0)
        tsne_layout.addWidget(self.tsne_features_label, 1, 1)
        tsne_layout.addWidget(QLabel("Best perplexity:"), 2, 0)
        tsne_layout.addWidget(self.tsne_perplexity_label, 2, 1)
        tsne_layout.addWidget(QLabel("KL divergence:"), 3, 0)
        tsne_layout.addWidget(self.tsne_kl_divergence_label, 3, 1)
        tsne_layout.addWidget(QLabel("Preprocessing applied:"), 4, 0)
        tsne_layout.addWidget(self.tsne_preprocessing_label, 4, 1)
        
        layout.addWidget(tsne_group)
        
        # t-SNE quality assessment
        quality_group = QGroupBox("t-SNE Quality Assessment")
        quality_layout = QVBoxLayout(quality_group)
        
        self.tsne_quality_text = QTextEdit()
        self.tsne_quality_text.setMaximumHeight(150)
        self.tsne_quality_text.setReadOnly(True)
        quality_layout.addWidget(self.tsne_quality_text)
        
        layout.addWidget(quality_group)
        
        return widget
        
    def create_feature_importance_tab(self) -> QWidget:
        """Create feature importance in PCA tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Feature importance table
        importance_group = QGroupBox("Feature Contributions to Principal Components")
        importance_layout = QVBoxLayout(importance_group)
        
        self.feature_importance_table = QTableWidget()
        self.feature_importance_table.setColumnCount(4)
        self.feature_importance_table.setHorizontalHeaderLabels(
            ["Component", "Feature", "Contribution", "Rank"]
        )
        importance_layout.addWidget(self.feature_importance_table)
        
        layout.addWidget(importance_group)
        
        return widget
        
    def create_visualizations_tab(self) -> QWidget:
        """Create visualizations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Visualization options
        viz_group = QGroupBox("Available Visualizations")
        viz_layout = QVBoxLayout(viz_group)
        
        self.viz_description = QTextEdit()
        self.viz_description.setMaximumHeight(200)
        self.viz_description.setReadOnly(True)
        viz_layout.addWidget(self.viz_description)
        
        # PCA Scree Plot
        scree_group = QGroupBox("PCA Scree Plot")
        scree_layout = QVBoxLayout(scree_group)
        
        self.scree_plot_widget = MatplotlibWidget()
        scree_layout.addWidget(self.scree_plot_widget)
        
        layout.addWidget(viz_group)
        layout.addWidget(scree_group)
        
        return widget
        
    def create_dim_recommendations_tab(self) -> QWidget:
        """Create dimensionality recommendations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.dim_recommendations_text = QTextEdit()
        self.dim_recommendations_text.setReadOnly(True)
        layout.addWidget(self.dim_recommendations_text)
        
        return widget
        
    def update_data(self, dimensionality_info: dict):
        """Update the dimensionality tab with analysis results"""
        # Update summary
        reduction_score = dimensionality_info.get('reduction_score', 0)
        self.reduction_score_label.setText(f"{reduction_score:.1f}/100")
        
        # Color code the score
        if reduction_score >= 80:
            color = "#4CAF50"  # Green
        elif reduction_score >= 60:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        self.reduction_score_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        data_prep = dimensionality_info.get('data_preparation', {})
        total_features = data_prep.get('processed_features', 0)
        self.total_features_label.setText(str(total_features))
        
        # Update PCA information
        pca_info = dimensionality_info.get('pca_analysis', {})
        if pca_info:
            self.update_pca_data(pca_info)
            
        # Update t-SNE information
        tsne_info = dimensionality_info.get('tsne_analysis', {})
        if tsne_info:
            self.update_tsne_data(tsne_info)
            
        # Update feature importance
        if 'feature_importance' in pca_info:
            self.update_feature_importance(pca_info['feature_importance'])
            
        # Update visualizations
        visualizations = dimensionality_info.get('visualizations', {})
        if visualizations:
            self.update_visualizations(visualizations)
            
        # Update recommendations
        recommendations = dimensionality_info.get('recommendations', [])
        self.dim_recommendations_text.setPlainText('\n'.join(recommendations))
        
    def update_pca_data(self, pca_info: dict):
        """Update PCA-specific data"""
        # Update summary labels
        comp_90 = pca_info.get('components_for_90_variance', 0)
        self.pca_components_label.setText(str(comp_90))
        
        # Update detailed PCA tab
        self.pca_components_90_label.setText(str(comp_90))
        
        comp_95 = pca_info.get('components_for_95_variance', 0)
        self.pca_components_95_label.setText(str(comp_95))
        
        total_var = pca_info.get('total_variance_explained', 0)
        self.pca_total_variance_label.setText(f"{total_var * 100:.2f}%")
        
        # Update variance table
        explained_var = pca_info.get('explained_variance_ratio', [])
        cumulative_var = pca_info.get('cumulative_variance', [])
        
        self.pca_variance_table.setRowCount(min(len(explained_var), 20))  # Show first 20 components
        
        for i, (var, cum_var) in enumerate(zip(explained_var[:20], cumulative_var[:20])):
            self.pca_variance_table.setItem(i, 0, QTableWidgetItem(f"PC{i+1}"))
            self.pca_variance_table.setItem(i, 1, QTableWidgetItem(f"{var:.4f}"))
            self.pca_variance_table.setItem(i, 2, QTableWidgetItem(f"{cum_var:.4f}"))
            self.pca_variance_table.setItem(i, 3, QTableWidgetItem(f"{var * 100:.2f}%"))
        
        self.pca_variance_table.resizeColumnsToContents()
        
    def update_tsne_data(self, tsne_info: dict):
        """Update t-SNE-specific data"""
        # Update summary and detailed info
        n_samples = tsne_info.get('n_samples_used', 0)
        n_features = tsne_info.get('n_features_used', 0)
        best_perp = tsne_info.get('best_perplexity', 0)
        preprocessing = tsne_info.get('preprocessing_applied', False)
        
        self.tsne_samples_label.setText(str(n_samples))
        self.tsne_features_label.setText(str(n_features))
        self.tsne_perplexity_label.setText(str(best_perp))
        self.tsne_preprocessing_label.setText("Yes (PCA)" if preprocessing else "No")
        
        # Get KL divergence for best perplexity
        perp_results = tsne_info.get('perplexity_results', {})
        if best_perp and best_perp in perp_results:
            kl_div = perp_results[best_perp]['kl_divergence']
            self.tsne_kl_divergence_label.setText(f"{kl_div:.4f}")
            
            # Update quality assessment
            if kl_div < 1.0:
                quality = "Excellent"
                color = "#4CAF50"
                description = "Perfect convergence. Clear clusters and patterns should be visible."
            elif kl_div < 2.0:
                quality = "Good"
                color = "#8BC34A"
                description = "Good convergence. Reliable visualization for pattern discovery."
            elif kl_div < 3.0:
                quality = "Fair"
                color = "#FF9800"
                description = "Moderate convergence. Try different perplexity values."
            else:
                quality = "Poor"
                color = "#F44336"
                description = "Poor convergence. Consider preprocessing or parameter tuning."
                
            self.tsne_quality_label.setText(quality)
            self.tsne_quality_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            quality_text = f"""t-SNE Quality Assessment:

Quality: {quality}
KL Divergence: {kl_div:.4f}
Perplexity: {best_perp}

{description}

Lower KL divergence indicates better convergence and more reliable visualization.
"""
            self.tsne_quality_text.setPlainText(quality_text)
        
    def update_feature_importance(self, feature_importance: dict):
        """Update feature importance table"""
        # Flatten the feature importance data
        importance_data = []
        for component, features in feature_importance.items():
            for rank, (feature, contribution) in enumerate(features, 1):
                importance_data.append({
                    'component': component,
                    'feature': feature,
                    'contribution': contribution,
                    'rank': rank
                })
        
        self.feature_importance_table.setRowCount(len(importance_data))
        
        for i, item in enumerate(importance_data):
            self.feature_importance_table.setItem(i, 0, QTableWidgetItem(item['component']))
            self.feature_importance_table.setItem(i, 1, QTableWidgetItem(item['feature']))
            self.feature_importance_table.setItem(i, 2, QTableWidgetItem(f"{item['contribution']:.4f}"))
            self.feature_importance_table.setItem(i, 3, QTableWidgetItem(str(item['rank'])))
        
        self.feature_importance_table.resizeColumnsToContents()
        
    def update_visualizations(self, visualizations: dict):
        """Update visualization displays"""
        # Update description
        description_text = """Available Visualizations:

• PCA Scree Plot: Shows explained variance by component
• PCA 2D Plot: First two principal components
• PCA 3D Plot: First three principal components  
• t-SNE 2D Plot: Non-linear dimensionality reduction
• t-SNE 3D Plot: 3D t-SNE embedding

Note: Interactive plots can be generated separately using the analysis results.
The plots help identify clusters, outliers, and data patterns.
"""
        self.viz_description.setPlainText(description_text)
        
        # Create and display scree plot
        scree_data = visualizations.get('pca_scree_plot', {})
        if scree_data:
            self.create_scree_plot(scree_data)
            
    def create_scree_plot(self, scree_data: dict):
        """Create PCA scree plot"""
        def plot_scree(figure, data):
            components = data.get('components', [])
            explained_var = data.get('explained_variance', [])
            cumulative_var = data.get('cumulative_variance', [])
            comp_80 = data.get('components_80', 0)
            comp_90 = data.get('components_90', 0)
            
            if not components or not explained_var:
                ax = figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No PCA scree plot data available', 
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Limit to first 20 components for readability
            components = components[:20]
            explained_var = explained_var[:20]
            cumulative_var = cumulative_var[:20]
            
            ax1 = figure.add_subplot(111)
            
            # Plot individual variance (bars)
            bars = ax1.bar(components, [v * 100 for v in explained_var], 
                          alpha=0.7, color='skyblue', label='Individual')
            
            # Plot cumulative variance (line)
            ax2 = ax1.twinx()
            line = ax2.plot(components, [v * 100 for v in cumulative_var], 
                           'ro-', color='red', label='Cumulative')
            
            # Add threshold lines
            if comp_80 <= len(components):
                ax1.axvline(x=comp_80, color='orange', linestyle='--', alpha=0.7, label='80% variance')
            if comp_90 <= len(components):
                ax1.axvline(x=comp_90, color='green', linestyle='--', alpha=0.7, label='90% variance')
            
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance (%)', color='blue')
            ax2.set_ylabel('Cumulative Variance (%)', color='red')
            ax1.set_title('PCA Scree Plot')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
            
            ax1.grid(True, alpha=0.3)
            figure.tight_layout()
        
        self.scree_plot_widget.plot(plot_scree, scree_data)


class ResultsTabWidget(QTabWidget):
    """Main results tab widget containing all analysis result tabs"""
    
    def __init__(self):
        super().__init__()
        self.init_tabs()
        
    def init_tabs(self):
        """Initialize all result tabs"""
        # Overview tab
        self.overview_tab = OverviewTab()
        self.addTab(self.overview_tab, "Overview")
        
        # Completeness tab
        self.completeness_tab = CompletenessTab()
        self.addTab(self.completeness_tab, "Data Completeness")
        
        # Distribution tab
        self.distribution_tab = DistributionTab()
        self.addTab(self.distribution_tab, "Data Distribution")
        
        # Data leakage tab
        self.leakage_tab = DataLeakageTab()
        self.addTab(self.leakage_tab, "Data Leakage")
        
        # Feature quality tab
        self.feature_quality_tab = FeatureQualityTab()
        self.addTab(self.feature_quality_tab, "Feature Quality")
        
        # Dimensionality reduction tab
        self.dimensionality_tab = DimensionalityTab()
        self.addTab(self.dimensionality_tab, "PCA & t-SNE")
        
    def update_results(self, results: dict):
        """Update all tabs with analysis results"""
        data_info = results.get('data_info', {})
        completeness_info = results.get('completeness', {})
        distribution_info = results.get('distribution', {})
        leakage_info = results.get('leakage', {})
        feature_quality_info = results.get('feature_quality', {})
        
        # Update overview
        self.overview_tab.update_data(data_info, completeness_info)
        
        # Update completeness if available
        if completeness_info:
            self.completeness_tab.update_data(completeness_info)
            
        # Update distribution if available
        if distribution_info:
            self.distribution_tab.update_data(distribution_info)
            
        # Update leakage detection if available
        if leakage_info:
            self.leakage_tab.update_data(leakage_info)
            
        # Update feature quality if available
        if feature_quality_info:
            self.feature_quality_tab.update_data(feature_quality_info)
            
        # Update dimensionality reduction if available
        dimensionality_info = results.get('dimensionality', {})
        if dimensionality_info:
            self.dimensionality_tab.update_data(dimensionality_info)