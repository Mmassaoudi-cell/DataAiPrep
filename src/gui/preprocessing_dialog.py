"""
Advanced preprocessing configuration dialog for DataAiPrep
PyCaret-inspired automated data preparation interface
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QPushButton, QTextEdit, QScrollArea, QFrame, QFormLayout, QSlider,
    QProgressBar, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import pandas as pd

from ..pipeline import DataAiPrepSetup
from ..core.data_loader import DataLoader


class PreprocessingWorker(QThread):
    """Worker thread for running preprocessing pipeline"""
    
    progress_updated = pyqtSignal(int, str)
    preprocessing_completed = pyqtSignal(object, object, dict)  # transformed_data, pipeline, metadata
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path: str, config: dict):
        super().__init__()
        self.file_path = file_path
        self.config = config
        
    def run(self):
        """Run the preprocessing in background thread"""
        try:
            # Step 1: Load data
            self.progress_updated.emit(10, "Loading data...")
            data_loader = DataLoader()
            df = data_loader.load_data(self.file_path)
            
            if df is None:
                self.error_occurred.emit("Failed to load data file")
                return
                
            # Step 2: Initialize setup
            self.progress_updated.emit(30, "Initializing preprocessing pipeline...")
            setup = DataAiPrepSetup()
            
            # Step 3: Run preprocessing
            self.progress_updated.emit(50, "Running automated preprocessing...")
            transformed_data, pipeline = setup.setup(
                data=df,
                **self.config
            )
            
            # Step 4: Generate metadata
            self.progress_updated.emit(90, "Generating metadata...")
            metadata = {
                'original_shape': df.shape,
                'transformed_shape': transformed_data.shape,
                'feature_metadata': setup.feature_metadata_,
                'conversion_report': getattr(pipeline.named_steps.get('preprocessor', {}).named_transformers_.get('num', {}), 'conversion_report_', {}),
                'config_used': self.config
            }
            
            self.progress_updated.emit(100, "Preprocessing complete!")
            self.preprocessing_completed.emit(transformed_data, pipeline, metadata)
            
        except Exception as e:
            self.error_occurred.emit(f"Preprocessing error: {str(e)}")


class AdvancedPreprocessingDialog(QDialog):
    """Advanced preprocessing configuration dialog with PyCaret-inspired interface"""
    
    def __init__(self, parent=None, file_path: str = None):
        super().__init__(parent)
        self.file_path = file_path
        self.preprocessing_worker = None
        self.original_data = None
        self.transformed_data = None
        self.pipeline = None
        self.metadata = None
        
        self.setWindowTitle("Advanced Data Preprocessing - PyCaret Style")
        self.setGeometry(100, 100, 1200, 800)
        self.setModal(True)
        
        self.init_ui()
        self.load_data_preview()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different configuration sections
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_data_overview_tab()
        self.create_type_handling_tab()
        self.create_preprocessing_tab()
        self.create_feature_engineering_tab()
        self.create_advanced_tab()
        self.create_results_tab()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Quick Preview")
        self.preview_btn.clicked.connect(self.run_preview)
        button_layout.addWidget(self.preview_btn)
        
        self.process_btn = QPushButton("Run Full Preprocessing")
        self.process_btn.clicked.connect(self.run_preprocessing)
        button_layout.addWidget(self.process_btn)
        
        self.export_btn = QPushButton("Export Pipeline")
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
    def create_data_overview_tab(self):
        """Create data overview and basic information tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Data information section
        info_group = QGroupBox("Dataset Information")
        info_layout = QFormLayout(info_group)
        
        self.file_info_label = QLabel("Loading...")
        info_layout.addRow("File:", self.file_info_label)
        
        self.shape_info_label = QLabel("Loading...")
        info_layout.addRow("Shape:", self.shape_info_label)
        
        self.memory_info_label = QLabel("Loading...")
        info_layout.addRow("Memory Usage:", self.memory_info_label)
        
        layout.addWidget(info_group)
        
        # Data preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.data_preview_table = QTableWidget()
        self.data_preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.data_preview_table)
        
        layout.addWidget(preview_group)
        
        # Type inference preview
        types_group = QGroupBox("Automatic Type Detection")
        types_layout = QVBoxLayout(types_group)
        
        self.type_detection_text = QTextEdit()
        self.type_detection_text.setMaximumHeight(150)
        self.type_detection_text.setPlainText("Run type detection to see results...")
        types_layout.addWidget(self.type_detection_text)
        
        detect_btn = QPushButton("Run Type Detection")
        detect_btn.clicked.connect(self.run_type_detection)
        types_layout.addWidget(detect_btn)
        
        layout.addWidget(types_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Data Overview")
        
    def create_type_handling_tab(self):
        """Create type handling configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Column type overrides
        override_group = QGroupBox("Manual Type Overrides")
        override_layout = QVBoxLayout(override_group)
        
        override_layout.addWidget(QLabel("Manually specify column types (overrides automatic detection):"))
        
        # Create lists for each type
        lists_layout = QHBoxLayout()
        
        # Numeric columns
        numeric_group = QGroupBox("Numeric Columns")
        numeric_layout = QVBoxLayout(numeric_group)
        self.numeric_list = QListWidget()
        numeric_layout.addWidget(self.numeric_list)
        lists_layout.addWidget(numeric_group)
        
        # Categorical columns
        categorical_group = QGroupBox("Categorical Columns")
        categorical_layout = QVBoxLayout(categorical_group)
        self.categorical_list = QListWidget()
        categorical_layout.addWidget(self.categorical_list)
        lists_layout.addWidget(categorical_group)
        
        # Datetime columns
        datetime_group = QGroupBox("Datetime Columns")
        datetime_layout = QVBoxLayout(datetime_group)
        self.datetime_list = QListWidget()
        datetime_layout.addWidget(self.datetime_list)
        lists_layout.addWidget(datetime_group)
        
        # Ignored columns
        ignored_group = QGroupBox("Ignored Columns")
        ignored_layout = QVBoxLayout(ignored_group)
        self.ignored_list = QListWidget()
        ignored_layout.addWidget(self.ignored_list)
        lists_layout.addWidget(ignored_group)
        
        override_layout.addLayout(lists_layout)
        layout.addWidget(override_group)
        
        # Type detection settings
        settings_group = QGroupBox("Type Detection Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.id_threshold_spin = QDoubleSpinBox()
        self.id_threshold_spin.setRange(0.1, 1.0)
        self.id_threshold_spin.setValue(0.9)
        self.id_threshold_spin.setSingleStep(0.1)
        settings_layout.addRow("ID Column Cardinality Threshold:", self.id_threshold_spin)
        
        self.numeric_threshold_spin = QDoubleSpinBox()
        self.numeric_threshold_spin.setRange(0.0, 0.5)
        self.numeric_threshold_spin.setValue(0.1)
        self.numeric_threshold_spin.setSingleStep(0.05)
        settings_layout.addRow("Numeric Conversion Error Tolerance:", self.numeric_threshold_spin)
        
        self.text_length_spin = QSpinBox()
        self.text_length_spin.setRange(10, 200)
        self.text_length_spin.setValue(50)
        settings_layout.addRow("Text vs Categorical Length Threshold:", self.text_length_spin)
        
        layout.addWidget(settings_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Type Handling")
        
    def create_preprocessing_tab(self):
        """Create basic preprocessing configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Target variable selection
        target_group = QGroupBox("Target Variable")
        target_layout = QFormLayout(target_group)
        
        self.target_combo = QComboBox()
        self.target_combo.addItem("None (Unsupervised)")
        target_layout.addRow("Target Column:", self.target_combo)
        
        layout.addWidget(target_group)
        
        # Missing value handling
        missing_group = QGroupBox("Missing Value Handling")
        missing_layout = QFormLayout(missing_group)
        
        self.imputation_combo = QComboBox()
        self.imputation_combo.addItems(["simple", "knn", "iterative"])
        missing_layout.addRow("Imputation Method:", self.imputation_combo)
        
        self.numeric_imputation_combo = QComboBox()
        self.numeric_imputation_combo.addItems(["mean", "median", "most_frequent", "constant"])
        missing_layout.addRow("Numeric Imputation:", self.numeric_imputation_combo)
        
        self.categorical_imputation_combo = QComboBox()
        self.categorical_imputation_combo.addItems(["most_frequent", "constant"])
        missing_layout.addRow("Categorical Imputation:", self.categorical_imputation_combo)
        
        layout.addWidget(missing_group)
        
        # Encoding
        encoding_group = QGroupBox("Categorical Encoding")
        encoding_layout = QFormLayout(encoding_group)
        
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["auto", "onehot", "ordinal", "target"])
        encoding_layout.addRow("Encoding Method:", self.encoding_combo)
        
        self.max_categories_spin = QSpinBox()
        self.max_categories_spin.setRange(2, 50)
        self.max_categories_spin.setValue(10)
        encoding_layout.addRow("Max Categories for OHE:", self.max_categories_spin)
        
        layout.addWidget(encoding_group)
        
        # Scaling and normalization
        scaling_group = QGroupBox("Scaling & Normalization")
        scaling_layout = QVBoxLayout(scaling_group)
        
        self.normalize_check = QCheckBox("Enable Normalization")
        scaling_layout.addWidget(self.normalize_check)
        
        normalize_method_layout = QFormLayout()
        self.normalize_method_combo = QComboBox()
        self.normalize_method_combo.addItems(["zscore", "minmax", "robust", "maxabs"])
        normalize_method_layout.addRow("Normalization Method:", self.normalize_method_combo)
        scaling_layout.addLayout(normalize_method_layout)
        
        layout.addWidget(scaling_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Preprocessing")
        
    def create_feature_engineering_tab(self):
        """Create feature engineering configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Feature creation
        creation_group = QGroupBox("Feature Creation")
        creation_layout = QVBoxLayout(creation_group)
        
        self.polynomial_check = QCheckBox("Polynomial Features")
        creation_layout.addWidget(self.polynomial_check)
        
        poly_layout = QFormLayout()
        self.polynomial_degree_spin = QSpinBox()
        self.polynomial_degree_spin.setRange(2, 5)
        self.polynomial_degree_spin.setValue(2)
        poly_layout.addRow("Polynomial Degree:", self.polynomial_degree_spin)
        creation_layout.addLayout(poly_layout)
        
        self.interaction_check = QCheckBox("Feature Interactions")
        creation_layout.addWidget(self.interaction_check)
        
        self.ratio_check = QCheckBox("Feature Ratios")
        creation_layout.addWidget(self.ratio_check)
        
        layout.addWidget(creation_group)
        
        # Feature selection
        selection_group = QGroupBox("Feature Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        self.feature_selection_check = QCheckBox("Enable Feature Selection")
        selection_layout.addWidget(self.feature_selection_check)
        
        selection_params_layout = QFormLayout()
        self.selection_threshold_spin = QDoubleSpinBox()
        self.selection_threshold_spin.setRange(0.1, 1.0)
        self.selection_threshold_spin.setValue(0.8)
        self.selection_threshold_spin.setSingleStep(0.1)
        selection_params_layout.addRow("Selection Threshold:", self.selection_threshold_spin)
        selection_layout.addLayout(selection_params_layout)
        
        layout.addWidget(selection_group)
        
        # Transformation
        transform_group = QGroupBox("Distribution Transformation")
        transform_layout = QVBoxLayout(transform_group)
        
        self.transformation_check = QCheckBox("Apply Transformations")
        transform_layout.addWidget(self.transformation_check)
        
        transform_params_layout = QFormLayout()
        self.transformation_method_combo = QComboBox()
        self.transformation_method_combo.addItems(["yeo-johnson", "box-cox", "quantile-normal", "quantile-uniform"])
        transform_params_layout.addRow("Transformation Method:", self.transformation_method_combo)
        transform_layout.addLayout(transform_params_layout)
        
        layout.addWidget(transform_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Feature Engineering")
        
    def create_advanced_tab(self):
        """Create advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Outlier handling
        outlier_group = QGroupBox("Outlier Handling")
        outlier_layout = QVBoxLayout(outlier_group)
        
        self.remove_outliers_check = QCheckBox("Remove Outliers")
        outlier_layout.addWidget(self.remove_outliers_check)
        
        outlier_params_layout = QFormLayout()
        self.outliers_method_combo = QComboBox()
        self.outliers_method_combo.addItems(["iqr", "isolation", "lof", "svm"])
        outlier_params_layout.addRow("Outlier Detection Method:", self.outliers_method_combo)
        
        self.outliers_threshold_spin = QDoubleSpinBox()
        self.outliers_threshold_spin.setRange(0.01, 0.3)
        self.outliers_threshold_spin.setValue(0.05)
        self.outliers_threshold_spin.setSingleStep(0.01)
        outlier_params_layout.addRow("Outlier Threshold:", self.outliers_threshold_spin)
        
        outlier_layout.addLayout(outlier_params_layout)
        layout.addWidget(outlier_group)
        
        # Multicollinearity
        multicollinearity_group = QGroupBox("Multicollinearity")
        multicollinearity_layout = QVBoxLayout(multicollinearity_group)
        
        self.remove_multicollinearity_check = QCheckBox("Remove Multicollinear Features")
        multicollinearity_layout.addWidget(self.remove_multicollinearity_check)
        
        multicollinearity_params_layout = QFormLayout()
        self.multicollinearity_threshold_spin = QDoubleSpinBox()
        self.multicollinearity_threshold_spin.setRange(0.7, 0.99)
        self.multicollinearity_threshold_spin.setValue(0.9)
        self.multicollinearity_threshold_spin.setSingleStep(0.05)
        multicollinearity_params_layout.addRow("Correlation Threshold:", self.multicollinearity_threshold_spin)
        
        multicollinearity_layout.addLayout(multicollinearity_params_layout)
        layout.addWidget(multicollinearity_group)
        
        # Imbalanced data
        imbalance_group = QGroupBox("Imbalanced Data")
        imbalance_layout = QVBoxLayout(imbalance_group)
        
        self.fix_imbalance_check = QCheckBox("Fix Class Imbalance")
        imbalance_layout.addWidget(self.fix_imbalance_check)
        
        imbalance_params_layout = QFormLayout()
        self.imbalance_method_combo = QComboBox()
        self.imbalance_method_combo.addItems(["smote", "adasyn", "borderline", "randomover", "randomunder"])
        imbalance_params_layout.addRow("Balancing Method:", self.imbalance_method_combo)
        
        imbalance_layout.addLayout(imbalance_params_layout)
        layout.addWidget(imbalance_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Advanced")
        
    def create_results_tab(self):
        """Create results and output tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results summary
        summary_group = QGroupBox("Preprocessing Results")
        summary_layout = QVBoxLayout(summary_group)
        
        self.results_text = QTextEdit()
        self.results_text.setPlainText("Run preprocessing to see results...")
        summary_layout.addWidget(self.results_text)
        
        layout.addWidget(summary_group)
        
        # Transformation comparison
        comparison_group = QGroupBox("Before vs After Comparison")
        comparison_layout = QHBoxLayout(comparison_group)
        
        # Before
        before_layout = QVBoxLayout()
        before_layout.addWidget(QLabel("Original Data"))
        self.before_table = QTableWidget()
        self.before_table.setMaximumHeight(200)
        before_layout.addWidget(self.before_table)
        comparison_layout.addLayout(before_layout)
        
        # After
        after_layout = QVBoxLayout()
        after_layout.addWidget(QLabel("Transformed Data"))
        self.after_table = QTableWidget()
        self.after_table.setMaximumHeight(200)
        after_layout.addWidget(self.after_table)
        comparison_layout.addLayout(after_layout)
        
        layout.addWidget(comparison_group)
        
        self.tab_widget.addTab(tab, "Results")
        
    def load_data_preview(self):
        """Load data preview for the overview tab"""
        if not self.file_path:
            return
            
        try:
            data_loader = DataLoader()
            df = data_loader.load_data_preview(self.file_path, n_rows=100)
            
            if df is not None:
                self.original_data = df
                
                # Update file info
                file_name = Path(self.file_path).name
                self.file_info_label.setText(file_name)
                self.shape_info_label.setText(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
                self.memory_info_label.setText(f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                # Update preview table
                self.update_preview_table(self.data_preview_table, df.head(10))
                
                # Update target combo
                self.target_combo.clear()
                self.target_combo.addItem("None (Unsupervised)")
                self.target_combo.addItems(list(df.columns))
                
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error loading data preview: {str(e)}")
            
    def update_preview_table(self, table: QTableWidget, df: pd.DataFrame):
        """Update a table widget with DataFrame data"""
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = str(df.iloc[i, j])
                if len(value) > 50:
                    value = value[:47] + "..."
                table.setItem(i, j, QTableWidgetItem(value))
                
        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
    def run_type_detection(self):
        """Run type detection preview"""
        if self.original_data is None:
            return
            
        try:
            from ..pipeline.setup import _TypeInferenceTransformer
            
            type_detector = _TypeInferenceTransformer(
                id_like_cardinality_threshold=self.id_threshold_spin.value(),
                numeric_conversion_threshold=self.numeric_threshold_spin.value(),
                text_length_threshold=self.text_length_spin.value()
            )
            
            type_detector.fit(self.original_data)
            
            report_text = "Automatic Type Detection Results:\n\n"
            report_text += f"Numeric Columns ({len(type_detector.numeric_columns_)}):\n"
            report_text += ", ".join(type_detector.numeric_columns_[:10])
            if len(type_detector.numeric_columns_) > 10:
                report_text += f" ... and {len(type_detector.numeric_columns_) - 10} more"
            report_text += "\n\n"
            
            report_text += f"Categorical Columns ({len(type_detector.categorical_columns_)}):\n"
            report_text += ", ".join(type_detector.categorical_columns_[:10])
            if len(type_detector.categorical_columns_) > 10:
                report_text += f" ... and {len(type_detector.categorical_columns_) - 10} more"
            report_text += "\n\n"
            
            report_text += f"Datetime Columns ({len(type_detector.datetime_columns_)}):\n"
            report_text += ", ".join(type_detector.datetime_columns_)
            report_text += "\n\n"
            
            report_text += f"Text Columns ({len(type_detector.text_columns_)}):\n"
            report_text += ", ".join(type_detector.text_columns_)
            report_text += "\n\n"
            
            report_text += f"ID-like Columns ({len(type_detector.id_columns_)}):\n"
            report_text += ", ".join(type_detector.id_columns_)
            
            self.type_detection_text.setPlainText(report_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error running type detection: {str(e)}")
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get the preprocessing configuration from UI"""
        config = {
            'target': self.target_combo.currentText() if self.target_combo.currentText() != "None (Unsupervised)" else None,
            'imputation_type': self.imputation_combo.currentText(),
            'numeric_imputation': self.numeric_imputation_combo.currentText(),
            'categorical_imputation': self.categorical_imputation_combo.currentText(),
            'encoding_method': self.encoding_combo.currentText(),
            'max_encoding_ohe': self.max_categories_spin.value(),
            'normalize': self.normalize_check.isChecked(),
            'normalize_method': self.normalize_method_combo.currentText(),
            'polynomial_features': self.polynomial_check.isChecked(),
            'polynomial_degree': self.polynomial_degree_spin.value(),
            'feature_interaction': self.interaction_check.isChecked(),
            'feature_ratio': self.ratio_check.isChecked(),
            'feature_selection': self.feature_selection_check.isChecked(),
            'feature_selection_threshold': self.selection_threshold_spin.value(),
            'transformation': self.transformation_check.isChecked(),
            'transformation_method': self.transformation_method_combo.currentText(),
            'remove_outliers': self.remove_outliers_check.isChecked(),
            'outliers_method': self.outliers_method_combo.currentText(),
            'outliers_threshold': self.outliers_threshold_spin.value(),
            'remove_multicollinearity': self.remove_multicollinearity_check.isChecked(),
            'multicollinearity_threshold': self.multicollinearity_threshold_spin.value(),
            'fix_imbalance': self.fix_imbalance_check.isChecked(),
            'fix_imbalance_method': self.imbalance_method_combo.currentText()
        }
        
        return config
    
    def run_preview(self):
        """Run a quick preprocessing preview"""
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "No data file selected")
            return
            
        config = self.get_preprocessing_config()
        config['html_report'] = False  # Disable report for quick preview
        
        self.start_preprocessing(config, preview_mode=True)
        
    def run_preprocessing(self):
        """Run full preprocessing"""
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "No data file selected")
            return
            
        config = self.get_preprocessing_config()
        self.start_preprocessing(config, preview_mode=False)
        
    def start_preprocessing(self, config: Dict[str, Any], preview_mode: bool = False):
        """Start the preprocessing worker thread"""
        self.process_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start preprocessing worker thread
        self.preprocessing_worker = PreprocessingWorker(self.file_path, config)
        self.preprocessing_worker.progress_updated.connect(self.on_progress_updated)
        self.preprocessing_worker.preprocessing_completed.connect(self.on_preprocessing_completed)
        self.preprocessing_worker.error_occurred.connect(self.on_preprocessing_error)
        self.preprocessing_worker.start()
        
    def on_progress_updated(self, progress: int, message: str):
        """Handle progress updates"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        
    def on_preprocessing_completed(self, transformed_data, pipeline, metadata):
        """Handle completed preprocessing"""
        self.transformed_data = transformed_data
        self.pipeline = pipeline
        self.metadata = metadata
        
        # Update results tab
        self.update_results_display()
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(5)  # Results tab
        
        # Reset UI
        self.process_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
    def on_preprocessing_error(self, error_message: str):
        """Handle preprocessing errors"""
        QMessageBox.critical(self, "Preprocessing Error", error_message)
        
        # Reset UI
        self.process_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
    def update_results_display(self):
        """Update the results display with preprocessing results"""
        if not self.metadata:
            return
            
        results_text = "Preprocessing Results Summary:\n\n"
        
        # Basic transformation info
        original_shape = self.metadata['original_shape']
        transformed_shape = self.metadata['transformed_shape']
        
        results_text += f"Data Transformation:\n"
        results_text += f"  Original: {original_shape[0]:,} rows × {original_shape[1]} columns\n"
        results_text += f"  Transformed: {transformed_shape[0]:,} rows × {transformed_shape[1]} columns\n"
        results_text += f"  Change: {transformed_shape[1] - original_shape[1]:+d} columns\n\n"
        
        # Feature metadata
        feature_meta = self.metadata['feature_metadata']
        results_text += f"Feature Classification:\n"
        results_text += f"  Numeric: {len(feature_meta.get('numeric_columns', []))}\n"
        results_text += f"  Categorical: {len(feature_meta.get('categorical_columns', []))}\n"
        results_text += f"  Datetime: {len(feature_meta.get('datetime_columns', []))}\n"
        results_text += f"  Text: {len(feature_meta.get('text_columns', []))}\n"
        results_text += f"  Ignored/ID-like: {len(feature_meta.get('ignored_columns', []))}\n\n"
        
        # Configuration used
        results_text += "Configuration Applied:\n"
        config = self.metadata['config_used']
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    results_text += f"  ✓ {key.replace('_', ' ').title()}\n"
            else:
                results_text += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        self.results_text.setPlainText(results_text)
        
        # Update comparison tables
        if self.original_data is not None:
            self.update_preview_table(self.before_table, self.original_data.head(10))
            
        if self.transformed_data is not None:
            display_data = self.transformed_data.head(10)
            # Handle array data from sklearn transformations
            if hasattr(display_data, 'values'):
                display_data = pd.DataFrame(display_data.values if hasattr(display_data, 'values') else display_data)
            self.update_preview_table(self.after_table, display_data)

