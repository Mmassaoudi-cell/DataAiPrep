"""
Configuration module for DataAiPrep
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class Config:
    """Application configuration manager"""
    
    def __init__(self):
        self.config_file = Path.home() / ".dataaiprep" / "config.json"
        self.default_config = {
            "ui": {
                "theme": "light",
                "window_size": [1400, 900],
                "window_position": [100, 100],
                "remember_last_file": True,
                "auto_analyze": False
            },
            "analysis": {
                "completeness": {
                    "missing_threshold": 5.0,
                    "high_missing_threshold": 50.0,
                    "enable_pattern_analysis": True
                },
                "distribution": {
                    "skewness_threshold": 1.0,
                    "outlier_methods": ["iqr", "zscore", "modified_zscore"],
                    "outlier_threshold": 10.0,
                    "normality_alpha": 0.05
                },
                "performance": {
                    "max_rows_pattern_analysis": 10000,
                    "sample_size_large_datasets": 100000,
                    "enable_parallel_processing": True
                }
            },
            "reports": {
                "default_format": "html",
                "include_plots": True,
                "auto_open_report": True,
                "report_directory": str(Path.home() / "DataAiPrep_Reports")
            },
            "data_loading": {
                "encoding_fallbacks": ["utf-8", "latin-1", "iso-8859-1", "cp1252"],
                "max_preview_rows": 1000,
                "auto_detect_types": True,
                "chunk_size": 10000
            }
        }
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.default_config, config)
            else:
                # Create default config file
                self.save_config(self.default_config)
                return self.default_config.copy()
        except Exception as e:
            logging.warning(f"Error loading config, using defaults: {str(e)}")
            return self.default_config.copy()
            
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file"""
        try:
            if config is None:
                config = self.config
                
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving config: {str(e)}")
            return False
            
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., 'ui.theme')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key_path: str, value: Any) -> bool:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., 'ui.theme')
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = key_path.split('.')
            config = self.config
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
                
            # Set the value
            config[keys[-1]] = value
            return True
        except Exception as e:
            logging.error(f"Error setting config value {key_path}: {str(e)}")
            return False
            
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults"""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
        
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        return self.save_config()
        
    def get_recent_files(self) -> list:
        """Get list of recently opened files"""
        return self.get('ui.recent_files', [])
        
    def add_recent_file(self, file_path: str) -> None:
        """Add file to recent files list"""
        recent_files = self.get_recent_files()
        
        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)
            
        # Add to beginning
        recent_files.insert(0, file_path)
        
        # Limit to 10 recent files
        recent_files = recent_files[:10]
        
        self.set('ui.recent_files', recent_files)
        self.save_config()


# Global config instance
config = Config()