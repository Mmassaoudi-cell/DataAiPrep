"""
Advanced Feature Engineering Module

Features:
- Automated interaction feature generation
- Polynomial features with optimal degree selection
- Date/time feature extraction (cyclical encoding)
- Text feature extraction (TF-IDF)
- Lag features for time-series
- Rolling statistics generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureInfo:
    """Container for feature information"""
    name: str
    original_features: List[str]
    feature_type: str
    importance_score: Optional[float]


class AdvancedFeatureEngineer:
    """
    Advanced automated feature engineering.
    
    Capabilities:
    - Interaction feature generation
    - Polynomial feature creation
    - Datetime feature extraction
    - Cyclical encoding
    - Text feature extraction
    - Time-series features (lag, rolling)
    - Feature importance evaluation
    """
    
    def __init__(
        self,
        max_interactions: int = 2,
        max_polynomial_degree: int = 3,
        max_lag: int = 5,
        rolling_windows: List[int] = None
    ):
        """
        Initialize the feature engineer.
        
        Args:
            max_interactions: Maximum interaction order
            max_polynomial_degree: Maximum polynomial degree
            max_lag: Maximum lag for time-series features
            rolling_windows: Window sizes for rolling statistics
        """
        self.max_interactions = max_interactions
        self.max_polynomial_degree = max_polynomial_degree
        self.max_lag = max_lag
        self.rolling_windows = rolling_windows or [3, 7, 14, 30]
        self.feature_info_ = {}
        
    def engineer_features(
        self,
        data: pd.DataFrame,
        target_column: str = None,
        datetime_columns: List[str] = None,
        text_columns: List[str] = None,
        time_series_columns: List[str] = None,
        interaction_columns: List[str] = None,
        polynomial_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive feature engineering.
        
        Args:
            data: Input DataFrame
            target_column: Target variable for importance evaluation
            datetime_columns: Columns to extract datetime features from
            text_columns: Columns to extract text features from
            time_series_columns: Columns for lag/rolling features
            interaction_columns: Columns for interaction features
            polynomial_columns: Columns for polynomial features
            
        Returns:
            Tuple of (engineered DataFrame, feature info dictionary)
        """
        result_df = data.copy()
        feature_report = {
            'original_features': list(data.columns),
            'datetime_features': [],
            'text_features': [],
            'interaction_features': [],
            'polynomial_features': [],
            'time_series_features': [],
            'total_new_features': 0
        }
        
        # Auto-detect datetime columns
        if datetime_columns is None:
            datetime_columns = self._detect_datetime_columns(data)
        
        # Extract datetime features
        for col in datetime_columns:
            if col in result_df.columns:
                dt_features = self._extract_datetime_features(result_df, col)
                result_df = pd.concat([result_df, dt_features], axis=1)
                feature_report['datetime_features'].extend(dt_features.columns.tolist())
        
        # Extract text features
        if text_columns:
            for col in text_columns:
                if col in result_df.columns:
                    text_features = self._extract_text_features(result_df, col)
                    result_df = pd.concat([result_df, text_features], axis=1)
                    feature_report['text_features'].extend(text_features.columns.tolist())
        
        # Generate interaction features
        if interaction_columns:
            interaction_features = self._generate_interactions(
                result_df, interaction_columns
            )
            result_df = pd.concat([result_df, interaction_features], axis=1)
            feature_report['interaction_features'].extend(
                interaction_features.columns.tolist()
            )
        
        # Generate polynomial features
        if polynomial_columns:
            poly_features = self._generate_polynomial_features(
                result_df, polynomial_columns, target_column
            )
            result_df = pd.concat([result_df, poly_features], axis=1)
            feature_report['polynomial_features'].extend(poly_features.columns.tolist())
        
        # Generate time-series features
        if time_series_columns:
            ts_features = self._generate_time_series_features(
                result_df, time_series_columns
            )
            result_df = pd.concat([result_df, ts_features], axis=1)
            feature_report['time_series_features'].extend(ts_features.columns.tolist())
        
        # Calculate feature importance if target is provided
        if target_column and target_column in data.columns:
            feature_report['feature_importance'] = self._calculate_feature_importance(
                result_df, target_column
            )
        
        feature_report['total_new_features'] = (
            len(feature_report['datetime_features']) +
            len(feature_report['text_features']) +
            len(feature_report['interaction_features']) +
            len(feature_report['polynomial_features']) +
            len(feature_report['time_series_features'])
        )
        
        feature_report['final_feature_count'] = len(result_df.columns)
        
        return result_df, feature_report
    
    def _detect_datetime_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect datetime columns"""
        datetime_cols = []
        
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                datetime_cols.append(col)
            elif data[col].dtype == object:
                # Try to parse as datetime
                try:
                    pd.to_datetime(data[col].dropna().head(100))
                    datetime_cols.append(col)
                except Exception:
                    continue
        
        return datetime_cols
    
    def _extract_datetime_features(
        self,
        data: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Extract comprehensive datetime features"""
        dt_series = pd.to_datetime(data[column], errors='coerce')
        
        features = pd.DataFrame(index=data.index)
        prefix = f"{column}_"
        
        # Basic components
        features[f"{prefix}year"] = dt_series.dt.year
        features[f"{prefix}month"] = dt_series.dt.month
        features[f"{prefix}day"] = dt_series.dt.day
        features[f"{prefix}hour"] = dt_series.dt.hour
        features[f"{prefix}dayofweek"] = dt_series.dt.dayofweek
        features[f"{prefix}quarter"] = dt_series.dt.quarter
        features[f"{prefix}weekofyear"] = dt_series.dt.isocalendar().week.astype(int)
        features[f"{prefix}is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
        features[f"{prefix}is_month_start"] = dt_series.dt.is_month_start.astype(int)
        features[f"{prefix}is_month_end"] = dt_series.dt.is_month_end.astype(int)
        
        # Cyclical encoding for periodic features
        # Month (12 cycle)
        features[f"{prefix}month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
        features[f"{prefix}month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)
        
        # Day of week (7 cycle)
        features[f"{prefix}dow_sin"] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
        features[f"{prefix}dow_cos"] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
        
        # Hour (24 cycle)
        if dt_series.dt.hour.max() > 0:
            features[f"{prefix}hour_sin"] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
            features[f"{prefix}hour_cos"] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
        
        # Day of year (365 cycle)
        features[f"{prefix}doy_sin"] = np.sin(2 * np.pi * dt_series.dt.dayofyear / 365)
        features[f"{prefix}doy_cos"] = np.cos(2 * np.pi * dt_series.dt.dayofyear / 365)
        
        return features
    
    def _extract_text_features(
        self,
        data: pd.DataFrame,
        column: str,
        max_features: int = 100
    ) -> pd.DataFrame:
        """Extract text features using TF-IDF"""
        text_data = data[column].fillna('').astype(str)
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names_out()
            
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"{column}_tfidf_{name}" for name in feature_names],
                index=data.index
            )
            
            # Add basic text statistics
            tfidf_df[f"{column}_length"] = text_data.str.len()
            tfidf_df[f"{column}_word_count"] = text_data.str.split().str.len()
            tfidf_df[f"{column}_avg_word_length"] = (
                text_data.str.len() / (text_data.str.split().str.len() + 1)
            )
            
            return tfidf_df
            
        except Exception:
            # Fallback to basic text features only
            features = pd.DataFrame(index=data.index)
            features[f"{column}_length"] = text_data.str.len()
            features[f"{column}_word_count"] = text_data.str.split().str.len()
            return features
    
    def _generate_interactions(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Generate interaction features"""
        numeric_cols = [c for c in columns if c in data.columns and 
                       pd.api.types.is_numeric_dtype(data[c])]
        
        if len(numeric_cols) < 2:
            return pd.DataFrame(index=data.index)
        
        features = pd.DataFrame(index=data.index)
        
        # Pairwise interactions
        for col1, col2 in combinations(numeric_cols, 2):
            # Multiplication
            features[f"{col1}_x_{col2}"] = data[col1] * data[col2]
            
            # Division (with safety)
            divisor = data[col2].replace(0, np.nan)
            features[f"{col1}_div_{col2}"] = data[col1] / divisor
            
            # Addition
            features[f"{col1}_plus_{col2}"] = data[col1] + data[col2]
            
            # Difference
            features[f"{col1}_minus_{col2}"] = data[col1] - data[col2]
        
        return features
    
    def _generate_polynomial_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        target_column: str = None
    ) -> pd.DataFrame:
        """Generate polynomial features with optimal degree selection"""
        numeric_cols = [c for c in columns if c in data.columns and 
                       pd.api.types.is_numeric_dtype(data[c])]
        
        if not numeric_cols:
            return pd.DataFrame(index=data.index)
        
        # Select optimal degree based on target correlation if available
        optimal_degree = 2  # Default
        
        if target_column and target_column in data.columns:
            optimal_degree = self._find_optimal_polynomial_degree(
                data[numeric_cols], data[target_column]
            )
        
        # Generate polynomial features
        poly = PolynomialFeatures(
            degree=min(optimal_degree, self.max_polynomial_degree),
            include_bias=False,
            interaction_only=False
        )
        
        # Scale data for numerical stability
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_cols].fillna(0))
        
        poly_features = poly.fit_transform(scaled_data)
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Filter out original and interaction features (keep only powers)
        poly_df = pd.DataFrame(
            poly_features,
            columns=feature_names,
            index=data.index
        )
        
        # Keep only polynomial terms (not original features)
        new_cols = [c for c in poly_df.columns if '^' in c]
        
        return poly_df[new_cols]
    
    def _find_optimal_polynomial_degree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_degree: int = 4
    ) -> int:
        """Find optimal polynomial degree based on mutual information"""
        best_degree = 1
        best_score = 0
        
        X_filled = X.fillna(0)
        y_filled = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else 0)
        
        for degree in range(1, max_degree + 1):
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly = poly.fit_transform(X_filled)
                
                # Calculate mutual information
                if y.nunique() <= 10:
                    mi_scores = mutual_info_classif(X_poly, y_filled)
                else:
                    mi_scores = mutual_info_regression(X_poly, y_filled)
                
                avg_mi = np.mean(mi_scores)
                
                if avg_mi > best_score:
                    best_score = avg_mi
                    best_degree = degree
                    
            except Exception:
                continue
        
        return best_degree
    
    def _generate_time_series_features(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Generate time-series features (lag and rolling)"""
        numeric_cols = [c for c in columns if c in data.columns and 
                       pd.api.types.is_numeric_dtype(data[c])]
        
        if not numeric_cols:
            return pd.DataFrame(index=data.index)
        
        features = pd.DataFrame(index=data.index)
        
        for col in numeric_cols:
            series = data[col]
            
            # Lag features
            for lag in range(1, min(self.max_lag + 1, len(data) // 10)):
                features[f"{col}_lag_{lag}"] = series.shift(lag)
            
            # Rolling statistics
            for window in self.rolling_windows:
                if window < len(data):
                    features[f"{col}_rolling_mean_{window}"] = series.rolling(
                        window=window, min_periods=1
                    ).mean()
                    features[f"{col}_rolling_std_{window}"] = series.rolling(
                        window=window, min_periods=1
                    ).std()
                    features[f"{col}_rolling_min_{window}"] = series.rolling(
                        window=window, min_periods=1
                    ).min()
                    features[f"{col}_rolling_max_{window}"] = series.rolling(
                        window=window, min_periods=1
                    ).max()
            
            # Differencing
            features[f"{col}_diff_1"] = series.diff(1)
            features[f"{col}_pct_change"] = series.pct_change()
            
            # Exponential moving average
            features[f"{col}_ewm_mean"] = series.ewm(span=7, min_periods=1).mean()
        
        return features
    
    def _calculate_feature_importance(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, float]:
        """Calculate feature importance using mutual information"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_column]
        
        if not numeric_cols:
            return {}
        
        X = data[numeric_cols].fillna(0)
        y = data[target_column].fillna(data[target_column].mode().iloc[0] 
            if len(data[target_column].mode()) > 0 else 0)
        
        try:
            if y.nunique() <= 10:
                mi_scores = mutual_info_classif(X, y)
            else:
                mi_scores = mutual_info_regression(X, y)
            
            importance = dict(zip(numeric_cols, mi_scores))
            return {k: float(v) for k, v in sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )}
        except Exception:
            return {}
    
    def generate_feature_selection_report(
        self,
        data: pd.DataFrame,
        target_column: str,
        n_features: int = 20
    ) -> Dict[str, Any]:
        """Generate feature selection report with multiple methods"""
        report = {
            'mutual_information': {},
            'correlation': {},
            'recommendations': []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_column]
        
        # Mutual Information
        report['mutual_information'] = self._calculate_feature_importance(
            data, target_column
        )
        
        # Correlation with target
        if target_column in data.columns:
            target = data[target_column]
            if pd.api.types.is_numeric_dtype(target):
                correlations = {}
                for col in numeric_cols:
                    try:
                        corr = data[col].corr(target)
                        correlations[col] = float(abs(corr))
                    except Exception:
                        continue
                report['correlation'] = dict(sorted(
                    correlations.items(), key=lambda x: x[1], reverse=True
                ))
        
        # Top features by both methods
        mi_top = set(list(report['mutual_information'].keys())[:n_features])
        corr_top = set(list(report['correlation'].keys())[:n_features])
        
        report['recommended_features'] = list(mi_top | corr_top)
        report['consensus_features'] = list(mi_top & corr_top)
        
        return report

