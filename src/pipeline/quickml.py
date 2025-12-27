from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import make_scorer, accuracy_score, r2_score


class QuickML:
    def compare_models(self, preprocessed_data: pd.DataFrame, target: str) -> pd.DataFrame:
        df = preprocessed_data.copy()
        if target not in df.columns:
            raise ValueError("Target column not found in preprocessed data")

        X = df.drop(columns=[target]).values
        y = df[target].values

        is_classification = not np.issubdtype(y.dtype, np.number) or len(np.unique(y)) <= 20

        if is_classification:
            models: List[tuple[str, Any]] = [
                ("LogisticRegression", LogisticRegression(max_iter=200)),
                ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=42)),
                ("KNN", KNeighborsClassifier(n_neighbors=5)),
                ("SVM", SVC()),
            ]
            scorer = make_scorer(accuracy_score)
        else:
            models = [
                ("Ridge", Ridge()),
                ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=42)),
                ("KNNRegressor", KNeighborsRegressor(n_neighbors=5)),
                ("SVR", SVR()),
            ]
            scorer = make_scorer(r2_score)

        rows = []
        for name, model in models:
            try:
                scores = cross_val_score(model, X, y, cv=3, scoring=scorer, n_jobs=None)
                rows.append({"model": name, "cv_mean": float(scores.mean()), "cv_std": float(scores.std())})
            except Exception as e:
                rows.append({"model": name, "cv_mean": float("nan"), "cv_std": float("nan"), "error": str(e)})
        return pd.DataFrame(rows)



