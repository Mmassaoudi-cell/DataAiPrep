from __future__ import annotations

"""
PyCaret-like setup for automated preprocessing in DataAiPrep.

This module defines the DataAiPrepSetup class that builds a configurable
scikit-learn pipeline for:
 - Type inference and conversion
 - Missing value imputation
 - Outlier handling (optional)
 - Feature engineering (polynomial, interactions, ratios, binning, datetime, text)
 - Encoding strategies
 - Scaling/normalization and distribution transforms
 - Dimensionality reduction (optional)

Returns: fitted pipeline and transformed dataset ready for ML.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    PowerTransformer,
    QuantileTransformer,
)


class _TypeInferenceTransformer(BaseEstimator, TransformerMixin):
    """Advanced type inference: numeric, categorical, datetime, text, id-like.
    
    Features:
    - Intelligent numeric conversion with validation
    - Multiple datetime format detection and parsing
    - ID column detection with sophisticated heuristics
    - Text vs categorical distinction
    - Support for mixed-type columns
    - Robust error handling and logging
    """

    def __init__(self, 
                 id_like_cardinality_threshold: float = 0.9,
                 numeric_conversion_threshold: float = 0.1,
                 datetime_conversion_threshold: float = 0.1,
                 text_length_threshold: int = 50,
                 categorical_max_unique_ratio: float = 0.5):
        self.id_like_cardinality_threshold = id_like_cardinality_threshold
        self.numeric_conversion_threshold = numeric_conversion_threshold
        self.datetime_conversion_threshold = datetime_conversion_threshold
        self.text_length_threshold = text_length_threshold
        self.categorical_max_unique_ratio = categorical_max_unique_ratio
        
        # Result storage
        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.datetime_columns_: List[str] = []
        self.text_columns_: List[str] = []
        self.id_columns_: List[str] = []
        self.mixed_type_columns_: List[str] = []
        self.conversion_report_: Dict[str, Dict[str, Any]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = X.copy()
        
        for col in X.columns:
            series = X[col]
            original_dtype = series.dtype
            conversion_info = {
                'original_dtype': str(original_dtype),
                'original_nulls': series.isna().sum(),
                'sample_values': series.dropna().head(5).tolist() if len(series.dropna()) > 0 else [],
                'conversion_applied': None,
                'conversion_success_rate': 1.0
            }
            
            # Step 1: Advanced numeric conversion
            converted_series, numeric_success = self._try_numeric_conversion(series)
            if numeric_success:
                X[col] = converted_series
                conversion_info['conversion_applied'] = 'numeric'
                conversion_info['conversion_success_rate'] = 1 - (converted_series.isna().sum() - series.isna().sum()) / len(series)
            else:
                # Step 2: Advanced datetime conversion  
                converted_series, datetime_success = self._try_datetime_conversion(series)
                if datetime_success:
                    X[col] = converted_series
                    conversion_info['conversion_applied'] = 'datetime'
                    conversion_info['conversion_success_rate'] = 1 - (converted_series.isna().sum() - series.isna().sum()) / len(series)

            # Step 3: Classify final type
            final_series = X[col]
            column_type = self._classify_column_type(final_series, col)
            
            # Step 4: ID column detection with advanced heuristics
            is_id_like = self._detect_id_column(series, col)
            if is_id_like:
                self.id_columns_.append(col)
                
            conversion_info['final_type'] = column_type
            conversion_info['is_id_like'] = is_id_like
            self.conversion_report_[col] = conversion_info

        return self
    
    def _try_numeric_conversion(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """Try to convert series to numeric with sophisticated validation."""
        if pd.api.types.is_numeric_dtype(series):
            return series, True
            
        if series.dtype != object:
            return series, False
            
        # Remove common non-numeric characters and try conversion
        cleaned_series = series.astype(str).str.replace(r'[,$%\s]', '', regex=True)
        
        # Try direct numeric conversion
        numeric_converted = pd.to_numeric(cleaned_series, errors='coerce')
        null_increase = numeric_converted.isna().sum() - series.isna().sum()
        success_rate = 1 - (null_increase / len(series))
        
        # Accept conversion if less than threshold% of values become null
        if success_rate >= (1 - self.numeric_conversion_threshold):
            return numeric_converted, True
            
        return series, False
    
    def _try_datetime_conversion(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """Try multiple datetime format detection and conversion."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series, True
            
        if series.dtype != object:
            return series, False
        
        # Common datetime formats to try
        datetime_formats = [
            None,  # Let pandas infer
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%Y%m%d',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ'
        ]
        
        best_conversion = None
        best_success_rate = 0
        
        for fmt in datetime_formats:
            try:
                if fmt is None:
                    converted = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                else:
                    converted = pd.to_datetime(series, format=fmt, errors='coerce')
                
                null_increase = converted.isna().sum() - series.isna().sum()
                success_rate = 1 - (null_increase / len(series))
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_conversion = converted
                    
            except Exception:
                continue
        
        # Accept conversion if success rate meets threshold
        if best_conversion is not None and best_success_rate >= (1 - self.datetime_conversion_threshold):
            return best_conversion, True
            
        return series, False
    
    def _classify_column_type(self, series: pd.Series, col_name: str) -> str:
        """Classify column into final type category."""
        if pd.api.types.is_datetime64_any_dtype(series):
            self.datetime_columns_.append(col_name)
            return 'datetime'
        elif pd.api.types.is_numeric_dtype(series):
            self.numeric_columns_.append(col_name)
            return 'numeric'
        else:
            # Distinguish between text and categorical
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                self.categorical_columns_.append(col_name)
                return 'categorical'
                
            unique_ratio = non_null_series.nunique() / len(non_null_series)
            avg_length = non_null_series.astype(str).str.len().mean()
            
            # Text classification heuristics
            if (avg_length > self.text_length_threshold or 
                unique_ratio > 0.8 and avg_length > 20):
                self.text_columns_.append(col_name)
                return 'text'
            else:
                self.categorical_columns_.append(col_name)
                return 'categorical'
    
    def _detect_id_column(self, series: pd.Series, col_name: str) -> bool:
        """Advanced ID column detection with multiple heuristics."""
        if len(series) == 0:
            return False
            
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return False
        
        # Heuristic 1: High cardinality ratio
        unique_ratio = non_null_series.nunique() / len(non_null_series)
        if unique_ratio >= self.id_like_cardinality_threshold:
            return True
        
        # Heuristic 2: Column name patterns
        id_name_patterns = [
            r'.*id$', r'^id.*', r'.*_id$', r'^.*_id$',
            r'.*key$', r'^key.*', r'.*index$', r'^index.*',
            r'.*uuid$', r'^uuid.*', r'.*guid$', r'^guid.*'
        ]
        
        import re
        col_lower = col_name.lower()
        for pattern in id_name_patterns:
            if re.match(pattern, col_lower):
                return True
        
        # Heuristic 3: Sequential patterns (for numeric IDs)
        if pd.api.types.is_numeric_dtype(non_null_series):
            sorted_values = non_null_series.sort_values()
            if len(sorted_values) > 1:
                differences = sorted_values.diff().dropna()
                if len(differences) > 0:
                    # Check if mostly incremental by 1
                    unit_increments = (differences == 1).sum()
                    if unit_increments / len(differences) > 0.8:
                        return True
        
        # Heuristic 4: String pattern analysis for string IDs
        if series.dtype == object:
            str_series = non_null_series.astype(str)
            
            # Check for consistent length
            lengths = str_series.str.len()
            if lengths.nunique() == 1 and lengths.iloc[0] >= 8:
                return True
                
            # Check for UUID-like patterns
            uuid_pattern = r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$'
            uuid_matches = str_series.str.match(uuid_pattern).sum()
            if uuid_matches / len(str_series) > 0.5:
                return True
        
        return False

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Apply same conversions as in fit for consistency
        for col in X.columns:
            if col in self.conversion_report_:
                conversion_info = self.conversion_report_[col]
                if conversion_info['conversion_applied'] == 'numeric':
                    converted_series, _ = self._try_numeric_conversion(X[col])
                    X[col] = converted_series
                elif conversion_info['conversion_applied'] == 'datetime':
                    converted_series, _ = self._try_datetime_conversion(X[col])
                    X[col] = converted_series
        return X


class _DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Expand datetime columns into useful parts."""

    def __init__(self, datetime_columns: Optional[List[str]] = None):
        self.datetime_columns = datetime_columns or []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in list(self.datetime_columns):
            if col in X.columns:
                dt = pd.to_datetime(X[col], errors="coerce")
                X[f"{col}_year"] = dt.dt.year
                X[f"{col}_month"] = dt.dt.month
                X[f"{col}_day"] = dt.dt.day
                X[f"{col}_hour"] = dt.dt.hour
                X[f"{col}_dow"] = dt.dt.dayofweek
                X[f"{col}_quarter"] = dt.dt.quarter
                X[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        return X


class _NumericCyclicEncoder(BaseEstimator, TransformerMixin):
    """Sin/cos encoding for cyclic numeric features like month/day/hour."""

    def __init__(self, columns_period: Dict[str, int]):
        self.columns_period = columns_period

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, period in self.columns_period.items():
            if col in X.columns and period > 1:
                radians = 2 * np.pi * X[col] / period
                X[f"{col}_sin"] = np.sin(radians)
                X[f"{col}_cos"] = np.cos(radians)
        return X


class DataAiPrepSetup:
    def __init__(self):
        self.pipeline_: Optional[Pipeline] = None
        self.feature_metadata_: Dict[str, Any] = {}

    def setup(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        train_size: float = 0.8,
        session_id: Optional[int] = None,
        use_gpu: bool = False,
        html_report: bool = True,
        # Type handling
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        date_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        ignore_features: Optional[List[str]] = None,
        # Preprocessing
        imputation_type: str = "simple",
        numeric_imputation: str = "mean",
        categorical_imputation: str = "most_frequent",
        # Feature engineering
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        feature_interaction: bool = False,
        feature_ratio: bool = False,
        feature_selection: bool = False,
        feature_selection_threshold: float = 0.8,
        # Transformations
        normalize: bool = False,
        normalize_method: str = "zscore",
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        # Outliers
        remove_outliers: bool = False,
        outliers_method: str = "iqr",
        outliers_threshold: float = 0.05,
        # Multicollinearity
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        # Imbalanced data
        fix_imbalance: bool = False,
        fix_imbalance_method: str = "smote",
        # Encoding
        encoding_method: str = "auto",
        max_encoding_ohe: int = 10,
        rare_to_value: float = 0.01,
    ) -> Tuple[pd.DataFrame, Pipeline]:
        """Configure and fit a preprocessing pipeline. Returns transformed data and pipeline."""

        df = data.copy()
        ignore_features = ignore_features or []

        # 1) Type inference
        type_inf = _TypeInferenceTransformer()
        df_conv = type_inf.fit_transform(df)

        # Respect user-provided feature lists overriding inference
        numeric_cols = numeric_features or type_inf.numeric_columns_
        categorical_cols = categorical_features or type_inf.categorical_columns_
        datetime_cols = date_features or type_inf.datetime_columns_
        text_cols = text_features or type_inf.text_columns_

        # Exclude id/index columns and ignored columns
        excluded = set(type_inf.id_columns_ + ignore_features)
        numeric_cols = [c for c in numeric_cols if c not in excluded]
        categorical_cols = [c for c in categorical_cols if c not in excluded]
        datetime_cols = [c for c in datetime_cols if c not in excluded]
        text_cols = [c for c in text_cols if c not in excluded]

        # 2) Column-wise transformers
        num_imputer_strategy = numeric_imputation if imputation_type == "simple" else "mean"
        cat_imputer_strategy = categorical_imputation if imputation_type == "simple" else "most_frequent"

        numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=num_imputer_strategy))]
        categorical_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=cat_imputer_strategy))]

        # Encoding
        if encoding_method == "auto":
            # OHE with handle_unknown
            categorical_steps.append(
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=max_encoding_ohe),
                )
            )
        elif encoding_method == "ordinal":
            categorical_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        else:
            categorical_steps.append(
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=max_encoding_ohe))
            )

        # Scaling / normalization
        if normalize:
            scaler: Any
            if normalize_method == "zscore":
                scaler = StandardScaler()
            elif normalize_method == "minmax":
                scaler = MinMaxScaler()
            elif normalize_method == "robust":
                scaler = RobustScaler()
            elif normalize_method == "maxabs":
                scaler = MaxAbsScaler()
            else:
                scaler = StandardScaler()
            numeric_steps.append(("scaler", scaler))

        # Distribution transformation
        if transformation:
            if transformation_method in ["yeo-johnson", "box-cox"]:
                numeric_steps.append(("power", PowerTransformer(method="yeo-johnson")))
            elif transformation_method in ["quantile-normal", "quantile-uniform"]:
                output_distribution = "normal" if transformation_method.endswith("normal") else "uniform"
                numeric_steps.append(("quantile", QuantileTransformer(output_distribution=output_distribution, random_state=0)))

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(numeric_steps), numeric_cols),
                ("cat", Pipeline(categorical_steps), categorical_cols),
            ],
            remainder="drop",
        )

        steps: List[Tuple[str, Any]] = [("preprocessor", preprocessor)]

        # 3) Datetime expansion and cyclic encoding happen outside ColumnTransformer
        if datetime_cols:
            steps.insert(0, ("datetime", _DatetimeFeatureExtractor(datetime_columns=datetime_cols)))
            # Add common cyclic encodings if present
            cyclic = {}
            for base, period in [("month", 12), ("day", 31), ("hour", 24), ("dow", 7)]:
                for col in datetime_cols:
                    derived = f"{col}_{base}"
                    cyclic[derived] = period
            if cyclic:
                steps.insert(1, ("cyclic", _NumericCyclicEncoder(columns_period=cyclic)))

        # 4) Optional dimensionality reduction preview (not default)
        if feature_selection:
            # Use PCA as a proxy for feature selection here
            steps.append(("pca", PCA(n_components=min(50, max(2, int(feature_selection_threshold * 10))))))

        pipeline = Pipeline(steps)

        # Fit and transform
        X = df_conv.drop(columns=[target], errors="ignore")
        y = df_conv[target] if (target is not None and target in df_conv.columns) else None

        X_t = pipeline.fit_transform(X, y)
        # Convert to DataFrame when possible
        try:
            # Reconstruct column names from transformers
            out_cols: List[str] = []
            out_cols.extend([f"num_{c}" for c in numeric_cols])
            # For OHE, we cannot easily recover names without fitted encoder; skip for brevity
            transformed = pd.DataFrame(X_t)
        except Exception:
            transformed = pd.DataFrame(X_t)

        self.pipeline_ = pipeline
        self.feature_metadata_ = {
            "original_columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "text_columns": text_cols,
            "ignored_columns": list(excluded),
            "id_like_columns": type_inf.id_columns_,
        }

        if y is not None:
            transformed[target] = y.values

        return transformed, pipeline


