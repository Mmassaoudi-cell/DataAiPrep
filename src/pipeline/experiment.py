from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fingerprint_df(df) -> str:
    try:
        sample = df.head(100).to_json(orient="split", date_format="iso")
        return hashlib.md5(sample.encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"


@dataclass
class ExperimentTracker:
    experiment_id: str
    timestamp: str
    configuration: Dict[str, Any]
    data_fingerprint: str
    transformations_applied: List[str]
    quality_scores: Dict[str, Any]
    processing_time: str
    memory_usage: str
    warnings: List[str]
    errors: List[str]

    @staticmethod
    def create(setup_config: Dict[str, Any], df) -> "ExperimentTracker":
        return ExperimentTracker(
            experiment_id=f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            configuration=setup_config,
            data_fingerprint=_fingerprint_df(df),
            transformations_applied=[],
            quality_scores={},
            processing_time="",
            memory_usage="",
            warnings=[],
            errors=[],
        )

    def save(self, output_path: str) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return str(p)



