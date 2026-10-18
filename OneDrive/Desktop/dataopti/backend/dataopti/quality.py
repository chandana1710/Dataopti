"""Deterministic data quality metrics derived from the uploaded dataframe."""

import numpy as np
import pandas as pd

from .validation import inspect_invalid_values
from .value_utils import stable_value


def score_data_quality(df, missing_count=None, duplicate_count=None, outlier_count=None, invalid_count=None):
    """Return an explainable 0-100 score and its measured components.

    Penalties use measured issue rates: missing cells (30%), duplicate rows
    (20%), IQR outliers (15%), invalid values (15%), constant columns (10%), and
    columns with more than half their values missing (10%).
    """
    rows, columns = df.shape
    cell_count = rows * columns
    missing_count = int(df.isna().sum().sum()) if missing_count is None else int(missing_count)
    if duplicate_count is None:
        try:
            duplicate_count = int(df.duplicated().sum())
        except (TypeError, ValueError):
            comparable = df.copy()
            for column in comparable.columns:
                if comparable[column].dtype == "object":
                    comparable[column] = comparable[column].map(stable_value)
            duplicate_count = int(comparable.duplicated().sum())

    numeric = df.select_dtypes(include=[np.number])
    outlier_count = _count_iqr_outliers(numeric) if outlier_count is None else int(outlier_count)
    invalid_count = sum(item["count"] for item in inspect_invalid_values(df)) if invalid_count is None else int(invalid_count)
    missing_rate = missing_count / cell_count if cell_count else 0.0
    duplicate_rate = duplicate_count / rows if rows else 0.0
    numeric_cell_count = rows * len(numeric.columns)
    outlier_rate = outlier_count / numeric_cell_count if numeric_cell_count else 0.0
    invalid_rate = invalid_count / cell_count if cell_count else 0.0
    constant_count = 0
    for column in df.columns:
        if df[column].isna().all():
            continue
        try:
            distinct = df[column].nunique(dropna=True)
        except TypeError:
            distinct = df[column].map(stable_value).nunique(dropna=True)
        constant_count += int(distinct <= 1)
    constant_rate = constant_count / columns if columns else 0.0
    high_missing_count = int((df.isna().mean() > 0.5).sum()) if rows else 0
    high_missing_rate = high_missing_count / columns if columns else 0.0

    components = {
        "missing_values": round(30 * min(missing_rate, 1.0), 2),
        "duplicate_rows": round(20 * min(duplicate_rate, 1.0), 2),
        "outliers": round(15 * min(outlier_rate, 1.0), 2),
        "invalid_values": round(15 * min(invalid_rate, 1.0), 2),
        "constant_columns": round(10 * min(constant_rate, 1.0), 2),
        "high_missing_columns": round(10 * min(high_missing_rate, 1.0), 2),
    }
    score = max(0, min(100, int(100 - sum(components.values()))))
    status = "Excellent" if score >= 90 else "Good" if score >= 75 else "Moderate" if score >= 50 else "Poor"

    explanations = []
    if missing_count:
        explanations.append(f"{missing_count:,} missing cell(s)")
    if duplicate_count:
        explanations.append(f"{duplicate_count:,} duplicate row(s)")
    if outlier_count:
        explanations.append(f"{outlier_count:,} numeric IQR outlier(s)")
    if invalid_count:
        explanations.append(f"{invalid_count:,} invalid numeric value(s)")
    if constant_count:
        explanations.append(f"{constant_count:,} constant column(s)")
    if high_missing_count:
        explanations.append(f"{high_missing_count:,} column(s) are more than half missing")

    return {
        "score": score,
        "status": status,
        "explanation": "No measured quality issues were found." if not explanations else "Quality score reflects " + ", ".join(explanations) + ".",
        "penalties": components,
        "metrics": {
            "missing_rate": round(missing_rate, 6),
            "duplicate_rate": round(duplicate_rate, 6),
            "numeric_outlier_rate": round(outlier_rate, 6),
            "invalid_value_rate": round(invalid_rate, 6),
            "constant_columns": constant_count,
            "columns_over_50_percent_missing": high_missing_count,
        },
    }


def aggregate_quality(profiles):
    """Combine per-table quality scores, weighting by the number of cells."""
    measured = [item for item in profiles if isinstance(item, dict) and item.get("data_quality")]
    if not measured:
        return None
    weights = [max(int(item.get("rows", 0)) * int(item.get("columns", 0)), 1) for item in measured]
    score = round(sum(item["data_quality"]["score"] * weight for item, weight in zip(measured, weights)) / sum(weights))
    status = "Excellent" if score >= 90 else "Good" if score >= 75 else "Moderate" if score >= 50 else "Poor"
    return {
        "score": score,
        "status": status,
        "dataset_count": len(measured),
        "explanation": f"Weighted from measured quality scores for {len(measured)} tabular dataset(s).",
    }


def _count_iqr_outliers(numeric):
    if numeric.empty:
        return 0
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    mask = (numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr))
    return int(mask.sum().sum())
