"""
Data loader module for supporting multiple data formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import sqlite3
import logging

# Optional imports for database support
try:
    from sqlalchemy import create_engine
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


class DataLoader:
    """Handles loading data from various formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {
            '.csv': self._load_csv,
            '.tsv': self._load_tsv,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.feather': self._load_feather,
            '.pkl': self._load_pickle,
            '.pickle': self._load_pickle
        }
        
    def load_data(self, file_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load data from file
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            DataFrame or None if loading failed
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return None
                
            extension = file_path.suffix.lower()
            
            if extension not in self.supported_formats:
                self.logger.error(f"Unsupported file format: {extension}")
                return None
                
            loader_func = self.supported_formats[extension]
            df = loader_func(file_path, **kwargs)
            
            if df is not None:
                self.logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None
            
    def load_data_preview(self, file_path: str, n_rows: int = 1000) -> Optional[pd.DataFrame]:
        """
        Load a preview of the data (first n rows)
        
        Args:
            file_path: Path to the data file
            n_rows: Number of rows to load for preview
            
        Returns:
            DataFrame preview or None if loading failed
        """
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension in ['.csv', '.tsv']:
                sep = '\t' if extension == '.tsv' else ','
                return pd.read_csv(file_path, nrows=n_rows, sep=sep)
            elif extension == '.xlsx' or extension == '.xls':
                return pd.read_excel(file_path, nrows=n_rows)
            elif extension == '.parquet':
                # For parquet, we'll load full file since it's columnar
                return self.load_data(file_path)
            elif extension == '.json':
                # For JSON, we'll try to load and limit rows
                df = self.load_data(file_path)
                return df.head(n_rows) if df is not None else None
            else:
                # Fall back to regular loading
                return self.load_data(file_path)
                
        except Exception as e:
            self.logger.error(f"Error loading data preview from {file_path}: {str(e)}")
            return None
            
    def _load_csv(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load CSV file"""
        default_kwargs = {
            'encoding': 'utf-8',
            'low_memory': False
        }
        default_kwargs.update(kwargs)
        
        try:
            return pd.read_csv(file_path, **default_kwargs)
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    default_kwargs['encoding'] = encoding
                    return pd.read_csv(file_path, **default_kwargs)
                except UnicodeDecodeError:
                    continue
            raise
            
    def _load_tsv(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load TSV file"""
        kwargs['sep'] = '\t'
        return self._load_csv(file_path, **kwargs)
        
    def _load_json(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load JSON file"""
        default_kwargs = {
            'orient': 'records'
        }
        default_kwargs.update(kwargs)
        
        try:
            # Try pandas read_json first
            return pd.read_json(file_path, **default_kwargs)
        except ValueError:
            # If that fails, try manual JSON loading
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle different JSON structures
                if 'data' in data:
                    return pd.DataFrame(data['data'])
                elif 'records' in data:
                    return pd.DataFrame(data['records'])
                else:
                    # Try to convert dict to DataFrame
                    return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported JSON structure: {type(data)}")
                
    def _load_parquet(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load Parquet file"""
        return pd.read_parquet(file_path, **kwargs)
        
    def _load_excel(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load Excel file"""
        default_kwargs = {
            'engine': 'openpyxl' if file_path.suffix == '.xlsx' else 'xlrd'
        }
        default_kwargs.update(kwargs)
        
        # If no sheet specified, load the first sheet
        if 'sheet_name' not in kwargs:
            # Get list of sheet names
            excel_file = pd.ExcelFile(file_path, engine=default_kwargs['engine'])
            if len(excel_file.sheet_names) > 1:
                self.logger.info(f"Multiple sheets found: {excel_file.sheet_names}. Loading first sheet.")
            default_kwargs['sheet_name'] = excel_file.sheet_names[0]
            
        return pd.read_excel(file_path, **default_kwargs)
        
    def _load_feather(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load Feather file"""
        return pd.read_feather(file_path, **kwargs)
        
    def _load_pickle(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load Pickle file"""
        return pd.read_pickle(file_path, **kwargs)
        
    def load_from_database(self, connection_string: str, query: str) -> Optional[pd.DataFrame]:
        """
        Load data from database
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            
        Returns:
            DataFrame or None if loading failed
        """
        if not HAS_SQLALCHEMY:
            self.logger.error("SQLAlchemy not available for database connections")
            return None
            
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            self.logger.info(f"Successfully loaded data from database: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from database: {str(e)}")
            return None
            
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a data file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(file_path)
            
            info = {
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix.lower(),
                'file_path': str(file_path),
                'supported': file_path.suffix.lower() in self.supported_formats
            }
            
            # Try to get more detailed info for supported formats
            if info['supported']:
                try:
                    df_preview = self.load_data_preview(str(file_path), n_rows=100)
                    if df_preview is not None:
                        info.update({
                            'estimated_rows': len(df_preview),  # This is just preview size
                            'columns': len(df_preview.columns),
                            'column_names': list(df_preview.columns),
                            'dtypes': df_preview.dtypes.to_dict(),
                            'has_header': True  # Assume yes for now
                        })
                except Exception:
                    # If preview fails, just return basic info
                    pass
                    
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {'error': str(e)}
            
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data and return basic statistics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation = {
                'is_valid': True,
                'issues': [],
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'column_types': df.dtypes.to_dict(),
                'has_duplicates': df.duplicated().any(),
                'duplicate_count': df.duplicated().sum(),
                'missing_data': df.isnull().sum().to_dict()
            }
            
            # Check for common issues
            if df.empty:
                validation['is_valid'] = False
                validation['issues'].append("DataFrame is empty")
                
            if df.shape[1] == 0:
                validation['is_valid'] = False
                validation['issues'].append("No columns found")
                
            # Check for columns with all missing values
            all_null_cols = [col for col, null_count in validation['missing_data'].items() 
                           if null_count == df.shape[0]]
            if all_null_cols:
                validation['issues'].append(f"Columns with all missing values: {all_null_cols}")
                
            # Check for very high missing data percentage
            total_cells = df.shape[0] * df.shape[1]
            total_missing = sum(validation['missing_data'].values())
            missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0
            
            if missing_percentage > 50:
                validation['issues'].append(f"High missing data percentage: {missing_percentage:.1f}%")
                
            # Check for unnamed columns
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            if unnamed_cols:
                validation['issues'].append(f"Unnamed columns detected: {len(unnamed_cols)}")
                
            return validation
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'issues': [f"Validation error: {str(e)}"]
            }
            
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_formats.keys())
        
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_formats