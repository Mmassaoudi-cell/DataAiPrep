"""
File drop widget for drag-and-drop functionality
"""

from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QPen, QFont


class FileDropWidget(QWidget):
    """Widget that accepts file drag and drop operations"""
    
    file_dropped = pyqtSignal(str)  # Emitted when a file is dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file_path = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the widget UI"""
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setMaximumHeight(120)
        
        layout = QVBoxLayout(self)
        
        # Main label
        self.main_label = QLabel("Drop data file here")
        self.main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.main_label.setFont(font)
        layout.addWidget(self.main_label)
        
        # Subtitle label
        self.subtitle_label = QLabel("Supported: CSV, JSON, Excel, Parquet")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("color: gray;")
        layout.addWidget(self.subtitle_label)
        
        # File path label (hidden initially)
        self.file_path_label = QLabel("")
        self.file_path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_path_label.setStyleSheet("color: blue; font-weight: bold;")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setVisible(False)
        layout.addWidget(self.file_path_label)
        
        self.update_style()
        
    def update_style(self):
        """Update widget styling"""
        if self.current_file_path:
            # File selected style
            self.setStyleSheet("""
                QWidget {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    background-color: #f0f8f0;
                }
            """)
        else:
            # Default style
            self.setStyleSheet("""
                QWidget {
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    background-color: #fafafa;
                }
                QWidget:hover {
                    border-color: #007ACC;
                    background-color: #f0f7ff;
                }
            """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if self.is_supported_file(file_path):
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        QWidget {
                            border: 2px solid #007ACC;
                            border-radius: 8px;
                            background-color: #e6f3ff;
                        }
                    """)
                    return
        
        event.ignore()
        
    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self.update_style()
        
    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if self.is_supported_file(file_path):
                    self.set_file_path(file_path)
                    self.file_dropped.emit(file_path)
                    event.acceptProposedAction()
                    return
        
        event.ignore()
        self.update_style()
        
    def is_supported_file(self, file_path: str) -> bool:
        """Check if the file format is supported"""
        supported_extensions = {'.csv', '.tsv', '.json', '.parquet', '.xlsx', '.xls'}
        return Path(file_path).suffix.lower() in supported_extensions
        
    def set_file_path(self, file_path: str):
        """Set the current file path and update UI"""
        self.current_file_path = file_path
        file_name = Path(file_path).name
        
        # Update labels
        self.main_label.setText("✓ File Selected")
        self.subtitle_label.setVisible(False)
        self.file_path_label.setText(file_name)
        self.file_path_label.setVisible(True)
        
        # Update styling
        self.update_style()
        
    def clear_file(self):
        """Clear the current file selection"""
        self.current_file_path = None
        
        # Reset labels
        self.main_label.setText("Drop data file here")
        self.subtitle_label.setVisible(True)
        self.file_path_label.setVisible(False)
        
        # Update styling
        self.update_style()