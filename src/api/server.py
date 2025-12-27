from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from ..pipeline import DataAiPrepSetup


app = FastAPI(title="DataAiPrep API", version="1.0.0")


class SetupConfig(BaseModel):
    target: Optional[str] = None
    normalize: bool = False
    normalize_method: str = "zscore"
    transformation: bool = False
    transformation_method: str = "yeo-johnson"
    encoding_method: str = "auto"


_last_pipeline = None
_last_transformed: Optional[pd.DataFrame] = None


@app.post("/setup")
async def setup_endpoint(config: SetupConfig, csv_path: Optional[str] = None) -> Dict[str, Any]:
    global _last_pipeline, _last_transformed
    if not csv_path:
        return {"error": "csv_path is required for this minimal endpoint"}
    df = pd.read_csv(csv_path)
    setup = DataAiPrepSetup()
    transformed, pipeline = setup.setup(
        data=df,
        target=config.target,
        normalize=config.normalize,
        normalize_method=config.normalize_method,
        transformation=config.transformation,
        transformation_method=config.transformation_method,
        encoding_method=config.encoding_method,
    )
    _last_pipeline = pipeline
    _last_transformed = transformed
    return {
        "rows": int(transformed.shape[0]),
        "cols": int(transformed.shape[1]),
        "feature_metadata": setup.feature_metadata_,
    }


@app.post("/transform")
async def transform_endpoint(csv_path: str) -> Dict[str, Any]:
    global _last_pipeline
    if _last_pipeline is None:
        return {"error": "No pipeline available. Call /setup first."}
    df = pd.read_csv(csv_path)
    X_t = _last_pipeline.transform(df)
    return {"rows": int(getattr(X_t, "shape", (len(df), 0))[0])}


@app.get("/health")
async def health():
    return {"status": "ok"}



