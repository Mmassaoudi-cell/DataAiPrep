from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


class PipelineExporter:
    def __init__(self, fitted_pipeline: Any, feature_metadata: Dict[str, Any]):
        self.pipeline = fitted_pipeline
        self.feature_metadata = feature_metadata or {}

    def export_pipeline(self, output_path: str, format: str = "joblib") -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "joblib":
            joblib.dump(self.pipeline, path)
        elif format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(self.pipeline, f)
        else:
            raise ValueError("Unsupported export format")
        return str(path)

    def generate_code(self, output_path: Optional[str] = None, language: str = "python") -> str:
        code = """
import pandas as pd
from sklearn.pipeline import Pipeline
from joblib import load

pipeline = load("PIPELINE_PATH")

def transform(df: pd.DataFrame) -> pd.DataFrame:
    X_t = pipeline.transform(df)
    try:
        return pd.DataFrame(X_t)
    except Exception:
        return pd.DataFrame(X_t)
"""
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(code)
        return code

    def export_feature_metadata(self, output_path: Optional[str] = None) -> pd.DataFrame:
        df = pd.DataFrame([
            {"key": k, "value": v} for k, v in self.feature_metadata.items()
        ])
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_json(p, orient="records", indent=2)
        return df



