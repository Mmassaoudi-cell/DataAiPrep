"""
Main window implementation for DataAiPrep
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTabWidget, QProgressBar, QTextEdit, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QSplitter, QFrame, QGridLayout,
    QComboBox, QCheckBox, QSpinBox, QMenuBar, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QByteArray
from PyQt6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent, QAction, QIcon, QPainter
import warnings

# Try to import SVG support (optional)
try:
    from PyQt6.QtSvg import QSvgRenderer
    from PyQt6.QtSvgWidgets import QSvgWidget
    HAS_SVG_SUPPORT = True
except ImportError:
    HAS_SVG_SUPPORT = False

# Suppress scientific computing warnings for cleaner GUI experience
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from ..core.data_loader import DataLoader
from ..analysis.completeness_analyzer import CompletenessAnalyzer
from ..analysis.distribution_analyzer import DistributionAnalyzer
from ..analysis.leakage_detector import LeakageDetector
from ..analysis.feature_quality_analyzer import FeatureQualityAnalyzer
from ..analysis.dimensionality_analyzer import DimensionalityAnalyzer
from ..reports.report_generator import ReportGenerator
from ..pipeline import DataAiPrepSetup
from .file_drop_widget import FileDropWidget
from .results_tabs import ResultsTabWidget
from .preprocessing_dialog import AdvancedPreprocessingDialog


class AnalysisWorker(QThread):
    """Worker thread for running data analysis"""
    
    progress_updated = pyqtSignal(int, str)
    analysis_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path: str, config: dict):
        super().__init__()
        self.file_path = file_path
        self.config = config
        self.data_loader = DataLoader()
        
    def run(self):
        """Run the analysis in background thread"""
        try:
            # Step 1: Load data
            self.progress_updated.emit(10, "Loading data...")
            df = self.data_loader.load_data(self.file_path)
            
            if df is None:
                self.error_occurred.emit("Failed to load data file")
                return
                
            # Step 2: Initialize analyzers
            self.progress_updated.emit(20, "Initializing analyzers...")
            results = {
                'data_info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
            }
            
            # Step 3: Completeness analysis
            if self.config.get('completeness_analysis', True):
                self.progress_updated.emit(40, "Analyzing data completeness...")
                completeness_analyzer = CompletenessAnalyzer()
                results['completeness'] = completeness_analyzer.analyze(df)
            
            # Step 4: Distribution analysis
            if self.config.get('distribution_analysis', True):
                self.progress_updated.emit(50, "Analyzing data distribution...")
                distribution_analyzer = DistributionAnalyzer()
                results['distribution'] = distribution_analyzer.analyze(df, self.config.get('target_variable'))
            
            # Step 5: Leakage detection
            if self.config.get('leakage_detection', True):
                self.progress_updated.emit(70, "Detecting data leakage...")
                leakage_detector = LeakageDetector()
                results['leakage'] = leakage_detector.analyze(df, self.config.get('target_variable'))
            
            # Step 6: Feature quality assessment
            if self.config.get('feature_quality_assessment', True):
                self.progress_updated.emit(75, "Assessing feature quality...")
                feature_quality_analyzer = FeatureQualityAnalyzer()
                results['feature_quality'] = feature_quality_analyzer.analyze(df, self.config.get('target_variable'))
            
            # Step 7: Dimensionality analysis (PCA & t-SNE)
            if self.config.get('dimensionality_analysis', True):
                self.progress_updated.emit(90, "Performing PCA & t-SNE analysis...")
                dimensionality_analyzer = DimensionalityAnalyzer()
                results['dimensionality'] = dimensionality_analyzer.analyze(df, self.config.get('target_variable'))
            
            # Step 8: Complete
            self.progress_updated.emit(100, "Analysis complete!")
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"Analysis error: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.analysis_results = None
        self.analysis_worker = None
        
        # Create menu bar
        self.create_menu_bar()
        
        # Set application icon
        self.set_application_icon()
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("DataAiPrep - ML Data Quality Assessment")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls and configuration
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results and visualizations
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
        # Status bar
        self.statusBar().showMessage("Ready to analyze data")
        
    def create_left_panel(self) -> QWidget:
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header with SVG Logo
        logo_path = Path(__file__).parent.parent.parent / "data_ai_prep_logo.svg"
        
        if HAS_SVG_SUPPORT and logo_path.exists():
            # Use SVG logo widget
            logo_widget = QSvgWidget(str(logo_path))
            logo_widget.setFixedSize(250, 60)  # Match SVG viewBox dimensions
            
            # Center the logo
            logo_container = QWidget()
            logo_layout = QHBoxLayout(logo_container)
            logo_layout.setContentsMargins(0, 10, 0, 20)
            logo_layout.addStretch()
            logo_layout.addWidget(logo_widget)
            logo_layout.addStretch()
            layout.addWidget(logo_container)
        else:
            # Fallback to text labels if SVG not available
            header_label = QLabel("DataAiPrep")
            header_font = QFont()
            header_font.setPointSize(16)
            header_font.setBold(True)
            header_label.setFont(header_font)
            header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header_label)
            
            subtitle_label = QLabel("ML Data Quality Assessment")
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle_label.setStyleSheet("color: gray; margin-bottom: 20px;")
            layout.addWidget(subtitle_label)
        
        # File import section
        import_group = QGroupBox("Data Import")
        import_layout = QVBoxLayout(import_group)
        
        # File drop widget
        self.file_drop_widget = FileDropWidget()
        self.file_drop_widget.file_dropped.connect(self.on_file_selected)
        import_layout.addWidget(self.file_drop_widget)
        
        # Browse button
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self.browse_files)
        import_layout.addWidget(browse_btn)
        
        layout.addWidget(import_group)
        
        # Configuration section
        config_group = QGroupBox("Analysis Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Target variable selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Variable:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        target_layout.addWidget(self.target_combo)
        config_layout.addLayout(target_layout)
        
        # Problem type selection
        problem_layout = QHBoxLayout()
        problem_layout.addWidget(QLabel("Problem Type:"))
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(["Auto-detect", "Classification", "Regression"])
        problem_layout.addWidget(self.problem_combo)
        config_layout.addLayout(problem_layout)
        
        # Analysis modules
        modules_label = QLabel("Analysis Modules:")
        modules_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        config_layout.addWidget(modules_label)
        
        self.completeness_check = QCheckBox("Data Completeness Analysis")
        self.completeness_check.setChecked(True)
        config_layout.addWidget(self.completeness_check)
        
        self.distribution_check = QCheckBox("Data Distribution Analysis")
        self.distribution_check.setChecked(True)
        config_layout.addWidget(self.distribution_check)
        
        self.leakage_check = QCheckBox("Data Leakage Detection")
        self.leakage_check.setChecked(True)
        config_layout.addWidget(self.leakage_check)
        
        self.feature_quality_check = QCheckBox("Feature Quality Assessment")
        self.feature_quality_check.setChecked(True)
        config_layout.addWidget(self.feature_quality_check)
        
        self.dimensionality_check = QCheckBox("PCA & t-SNE Analysis")
        self.dimensionality_check.setChecked(True)
        config_layout.addWidget(self.dimensionality_check)
        
        layout.addWidget(config_group)
        
        # Analysis controls
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Start analysis button
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_analysis)
        controls_layout.addWidget(self.analyze_btn)
        
        # Advanced preprocessing button
        self.preprocess_btn = QPushButton("Advanced Preprocessing")
        self.preprocess_btn.setEnabled(False)
        self.preprocess_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E5F8B;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.preprocess_btn.clicked.connect(self.show_advanced_preprocessing_dialog)
        controls_layout.addWidget(self.preprocess_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        controls_layout.addWidget(self.progress_label)
        
        layout.addWidget(controls_group)
        
        # Quick insights panel
        insights_group = QGroupBox("Quick Insights")
        insights_layout = QVBoxLayout(insights_group)
        
        self.insights_text = QTextEdit()
        self.insights_text.setMaximumHeight(150)
        self.insights_text.setPlainText("Load a dataset to see quick insights...")
        insights_layout.addWidget(self.insights_text)
        
        layout.addWidget(insights_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_right_panel(self) -> QWidget:
        """Create the right results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Results header
        results_label = QLabel("Analysis Results")
        results_font = QFont()
        results_font.setPointSize(14)
        results_font.setBold(True)
        results_label.setFont(results_font)
        layout.addWidget(results_label)
        
        # Results tabs
        self.results_tabs = ResultsTabWidget()
        layout.addWidget(self.results_tabs)
        
        return panel
        
    def setup_connections(self):
        """Setup signal connections"""
        pass
        
    def browse_files(self):
        """Open file browser dialog"""
        file_filter = "Data Files (*.csv *.tsv *.json *.parquet *.xlsx *.xls);;CSV Files (*.csv);;JSON Files (*.json);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet);;All Files (*)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            file_filter
        )
        
        if file_path:
            self.on_file_selected(file_path)
            
    def on_file_selected(self, file_path: str):
        """Handle file selection"""
        self.current_file = file_path
        
        # Update UI
        self.file_drop_widget.set_file_path(file_path)
        self.analyze_btn.setEnabled(True)
        self.preprocess_btn.setEnabled(True)
        
        # Load data preview and update target variable options
        try:
            data_loader = DataLoader()
            df_preview = data_loader.load_data_preview(file_path)
            if df_preview is not None:
                self.target_combo.clear()
                self.target_combo.addItem("None (Unsupervised)")
                self.target_combo.addItems(list(df_preview.columns))
                self.target_combo.setEnabled(True)
                
                # Update quick insights
                insights = f"""File: {Path(file_path).name}
Shape: {df_preview.shape[0]:,} rows × {df_preview.shape[1]} columns
Memory: {df_preview.memory_usage(deep=True).sum() / 1024**2:.1f} MB

🔧 Ready for analysis and preprocessing!
💡 Try the Advanced Preprocessing for PyCaret-style automated data preparation."""
                self.insights_text.setPlainText(insights)
                
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error loading file preview: {str(e)}")
            
        self.statusBar().showMessage(f"File selected: {Path(file_path).name}")
        
    def start_analysis(self):
        """Start the data analysis process"""
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "Please select a data file first.")
            return
            
        # Prepare configuration
        config = {
            'target_variable': self.target_combo.currentText() if self.target_combo.currentText() != "None (Unsupervised)" else None,
            'problem_type': self.problem_combo.currentText(),
            'completeness_analysis': self.completeness_check.isChecked(),
            'distribution_analysis': self.distribution_check.isChecked(),
            'leakage_detection': self.leakage_check.isChecked(),
            'feature_quality_assessment': self.feature_quality_check.isChecked(),
            'dimensionality_analysis': self.dimensionality_check.isChecked()
        }
        
        # Update UI for analysis state
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start analysis worker thread
        self.analysis_worker = AnalysisWorker(self.current_file, config)
        self.analysis_worker.progress_updated.connect(self.on_progress_updated)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.start()
        
    def on_progress_updated(self, progress: int, message: str):
        """Handle progress updates"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        
    def on_analysis_completed(self, results: dict):
        """Handle completed analysis"""
        self.analysis_results = results
        
        # Update results tabs
        self.results_tabs.update_results(results)
        
        # Update quick insights
        data_info = results.get('data_info', {})
        completeness_info = results.get('completeness', {})
        
        insights = f"""Analysis Complete!

Dataset: {data_info.get('shape', ['?', '?'])[0]:,} rows × {data_info.get('shape', ['?', '?'])[1]} columns
Memory: {data_info.get('memory_usage', 0) / 1024**2:.1f} MB

Missing Data: {completeness_info.get('overall_missing_percentage', 0):.1f}%
Complete Columns: {completeness_info.get('complete_columns_count', 0)}
Issues Found: {len(completeness_info.get('issues', []))}"""
        
        self.insights_text.setPlainText(insights)
        
        # Reset UI
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.statusBar().showMessage("Analysis completed successfully")
        
    def on_analysis_error(self, error_message: str):
        """Handle analysis errors"""
        QMessageBox.critical(self, "Analysis Error", error_message)
        
        # Reset UI
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.statusBar().showMessage("Analysis failed")
        
    def set_application_icon(self):
        """Set the application icon from SVG logo"""
        try:
            logo_path = Path(__file__).parent.parent.parent / "data_ai_prep_logo.svg"
            
            if HAS_SVG_SUPPORT and logo_path.exists():
                # Load SVG and convert to icon
                renderer = QSvgRenderer(str(logo_path))
                
                # Create pixmap for icon
                pixmap = QPixmap(64, 64)
                pixmap.fill(Qt.GlobalColor.transparent)
                
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                
                # Set as window icon
                icon = QIcon(pixmap)
                self.setWindowIcon(icon)
                
                # Also set for the application
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().setWindowIcon(icon)
                
                print("✅ DataAiPrep logo loaded successfully!")
                
            else:
                print("ℹ️ SVG logo not found or SVG support unavailable, using default icon")
                
        except Exception as e:
            print(f"⚠️ Could not load application icon: {str(e)}")
        
    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About DataAiPrep', self)
        about_action.setStatusTip('About DataAiPrep')
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        # Advanced preprocessing dialog
        advanced_preprocess_action = QAction('Advanced Preprocessing...', self)
        advanced_preprocess_action.setStatusTip('Open advanced PyCaret-style preprocessing interface')
        advanced_preprocess_action.triggered.connect(self.show_advanced_preprocessing_dialog)
        tools_menu.addAction(advanced_preprocess_action)
        
        # Quick preprocessing preview
        preprocess_action = QAction('Quick Auto-Preprocess', self)
        preprocess_action.setStatusTip('Run automated preprocessing preview')
        preprocess_action.triggered.connect(self.run_auto_preprocess_preview)
        tools_menu.addAction(preprocess_action)
        
    def show_about_dialog(self):
        """Show the About dialog"""
        dialog = AboutDialog(self)
        dialog.exec()
        
    def show_advanced_preprocessing_dialog(self):
        """Show the advanced preprocessing dialog"""
        if not self.current_file:
            QMessageBox.information(self, "Advanced Preprocessing", "Please select a data file first.")
            return
            
        dialog = AdvancedPreprocessingDialog(self, self.current_file)
        dialog.exec()

    def run_auto_preprocess_preview(self):
        """Run a quick preprocessing setup on the currently selected file and show a brief preview."""
        if not self.current_file:
            QMessageBox.information(self, "Auto-Preprocess", "Please select a data file first.")
            return
        try:
            loader = DataLoader()
            df = loader.load_data(self.current_file)
            if df is None or df.empty:
                QMessageBox.warning(self, "Auto-Preprocess", "Failed to load data for preprocessing.")
                return

            setup = DataAiPrepSetup()
            transformed, pipeline = setup.setup(
                data=df,
                target=self.target_combo.currentText() if self.target_combo.currentText() != "None (Unsupervised)" else None,
                normalize=True,
                normalize_method='zscore',
                transformation=False,
                encoding_method='auto',
            )

            preview = transformed.head(5)
            self.insights_text.setPlainText(
                f"Auto-Preprocess Preview (Beta)\n\n"
                f"Original shape: {df.shape}\n"
                f"Transformed shape: {transformed.shape}\n"
                f"Numeric: {len(setup.feature_metadata_.get('numeric_columns', []))}, "
                f"Categorical: {len(setup.feature_metadata_.get('categorical_columns', []))}, "
                f"Datetime: {len(setup.feature_metadata_.get('datetime_columns', []))}\n"
                f"Ignored/ID-like: {len(setup.feature_metadata_.get('ignored_columns', []))}\n\n"
                f"Preview (first 5 rows):\n{preview.to_string(index=False)[:2000]}"
            )
            self.statusBar().showMessage("Auto-Preprocess preview complete")
        except Exception as e:
            QMessageBox.critical(self, "Auto-Preprocess Error", str(e))
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Analysis in Progress",
                "Analysis is currently running. Do you want to quit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.analysis_worker.terminate()
                self.analysis_worker.wait(3000)  # Wait up to 3 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class AboutDialog(QDialog):
    """About dialog showing application and author information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About DataAiPrep")
        self.setFixedSize(600, 500)  # Increased size for better content display
        self.setModal(True)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the About dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add logo if available
        self.add_logo(layout)
        
        # Application title
        title_label = QLabel("DataAiPrep")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2E86AB; padding: 5px;")
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("ML Data Quality Assessment Tool")
        subtitle_font = QFont()
        subtitle_font.setPointSize(11)
        subtitle_font.setItalic(True)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #666; padding: 2px;")
        layout.addWidget(subtitle_label)
        
        # Version info
        version_label = QLabel("Version 1.0.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #888; padding: 2px;")
        layout.addWidget(version_label)
        
        # Add some space
        layout.addSpacing(10)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Add some space
        layout.addSpacing(5)
        
        # Developer information
        dev_title = QLabel("Developed by")
        dev_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dev_title_font = QFont()
        dev_title_font.setBold(True)
        dev_title.setFont(dev_title_font)
        dev_title.setStyleSheet("color: #333; padding: 2px;")
        layout.addWidget(dev_title)
        
        # Author name
        author_label = QLabel("Mohamed Massaoudi, PhD")
        author_font = QFont()
        author_font.setPointSize(13)
        author_font.setBold(True)
        author_label.setFont(author_font)
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet("color: #2E86AB; padding: 3px;")
        layout.addWidget(author_label)
        
        # Title
        title_info = QLabel("Senior Research Engineer")
        title_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_info.setStyleSheet("color: #555; padding: 2px;")
        layout.addWidget(title_info)
        
        # Department
        dept_info = QLabel("Electrical and Computer Engineering")
        dept_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dept_info.setStyleSheet("color: #555; padding: 2px;")
        layout.addWidget(dept_info)
        
        # University
        uni_info = QLabel("Texas A&M University")
        uni_font = QFont()
        uni_font.setBold(True)
        uni_info.setFont(uni_font)
        uni_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        uni_info.setStyleSheet("color: #500000; padding: 3px;")  # Aggie maroon
        layout.addWidget(uni_info)
        
        # Add some space
        layout.addSpacing(5)
        
        # Email
        email_label = QLabel("📧 mohamed.massaoudi@tamu.edu")
        email_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        email_label.setStyleSheet("color: #2E86AB; padding: 2px;")
        layout.addWidget(email_label)
        
        # Add some space
        layout.addSpacing(5)
        
        # Research Affiliations
        affiliations_title = QLabel("Research Affiliations")
        affiliations_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        affiliations_font = QFont()
        affiliations_font.setBold(True)
        affiliations_title.setFont(affiliations_font)
        affiliations_title.setStyleSheet("color: #333; padding: 2px;")
        layout.addWidget(affiliations_title)
        
        # Affiliation 1
        lab1_label = QLabel("🏛️ Resilient Energy Systems Lab (RESLab)")
        lab1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab1_label.setStyleSheet("color: #555; padding: 1px;")
        layout.addWidget(lab1_label)
        
        # Affiliation 2
        lab2_label = QLabel("🔬 TEES Smart Grid Center")
        lab2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab2_label.setStyleSheet("color: #555; padding: 1px;")
        layout.addWidget(lab2_label)
        
        # Add some space
        layout.addSpacing(5)
        
        # Address
        address_label = QLabel("📍 3128 TAMU, College Station, TX 77843-3128")
        address_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        address_label.setStyleSheet("color: #777; font-size: 10px; padding: 2px;")
        layout.addWidget(address_label)
        
        # Add some space
        layout.addSpacing(10)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setFixedSize(100, 30)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E5F8B;
            }
        """)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def add_logo(self, layout):
        """Add the DataAiPrep logo to the dialog"""
        try:
            logo_path = Path(__file__).parent.parent.parent / "data_ai_prep_logo.svg"
            
            if HAS_SVG_SUPPORT and logo_path.exists():
                # Create SVG widget for the logo
                logo_widget = QSvgWidget(str(logo_path))
                logo_widget.setFixedSize(200, 48)  # Scale the logo appropriately
                
                # Center the logo
                logo_layout = QHBoxLayout()
                logo_layout.addStretch()
                logo_layout.addWidget(logo_widget)
                logo_layout.addStretch()
                
                layout.addLayout(logo_layout)
                layout.addSpacing(5)
                
            else:
                # Fallback: just add some spacing where logo would be
                layout.addSpacing(5)
                
        except Exception as e:
            # Graceful fallback - just add spacing
            layout.addSpacing(5)