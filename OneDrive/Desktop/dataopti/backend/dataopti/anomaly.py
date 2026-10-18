"""Measured row-level anomaly detection for numeric tabular signals."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, max_rows=50000, random_state=42):
    """Use Isolation Forest on usable numeric columns and return a compact report."""
    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if len(df) < 8 or numeric.empty:
        return {
            "status": "unavailable",
            "method": "Isolation Forest",
            "rows_analyzed": int(len(df)),
            "anomaly_count": 0,
            "anomaly_percentage": 0.0,
            "columns_analyzed": [str(column) for column in numeric.columns],
            "reason": "At least eight rows and one varying numeric column are needed for anomaly detection.",
        }

    sampled = numeric.sample(n=max_rows, random_state=random_state) if len(numeric) > max_rows else numeric
    positions = df.index.get_indexer(sampled.index)
    medians = sampled.median().fillna(0)
    values = sampled.fillna(medians).to_numpy(dtype=float)
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        max_samples=min(256, len(values)),
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(values)
    anomalous = labels == -1
    anomaly_count = int(anomalous.sum())
    rows_analyzed = int(len(values))
    return {
        "status": "complete",
        "method": "Isolation Forest",
        "rows_analyzed": rows_analyzed,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(anomaly_count / max(rows_analyzed, 1) * 100, 2),
        "columns_analyzed": [str(column) for column in sampled.columns],
        "sampled": rows_analyzed < len(df),
        "anomaly_row_positions": positions[anomalous][:100].astype(int).tolist(),
        "scores_mean": round(float(np.mean(model.score_samples(values))), 6),
        "explanation": "Isolation Forest flagged rows with unusual combinations of numeric values; flagged rows are reported, not removed.",
    }
