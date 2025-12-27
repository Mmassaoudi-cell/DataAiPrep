from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class IntelligentAdvisor:
    def analyze_and_recommend(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        recs = {"critical": [], "important": [], "optional": []}

        # Missing values
        missing_pct = data.isna().mean().sort_values(ascending=False)
        for col, pct in missing_pct.items():
            if pct >= 0.2:
                recs["critical"].append(f"Column '{col}' has {pct*100:.1f}% missing. Recommend advanced imputation or removal.")
            elif pct > 0:
                recs["important"].append(f"Column '{col}' has {pct*100:.1f}% missing. Consider simple imputation.")

        # Skewness
        for col in data.select_dtypes(include=[np.number]).columns:
            ser = data[col].dropna()
            if len(ser) >= 10:
                skew = ser.skew()
                if abs(skew) > 1.0:
                    recs["important"].append(
                        f"Feature '{col}' is highly skewed (skewness={skew:.2f}). Consider log/yeo-johnson."
                    )

        # High correlation pairs
        num = data.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            corr = num.corr().abs()
            upper = corr.where(~np.tril(np.ones(corr.shape, dtype=bool)))
            high_pairs = upper.stack().loc[lambda s: s >= 0.95]
            for (f1, f2), v in high_pairs.items():
                recs["important"].append(
                    f"High correlation ({v:.2f}) between '{f1}' and '{f2}'. Consider removing one."
                )

        # Potential id-like columns
        for col in data.columns:
            nunique = data[col].nunique(dropna=True)
            if nunique / max(1, len(data)) > 0.9:
                recs["optional"].append(f"Column '{col}' looks like an identifier. Exclude from modeling.")

        return recs



