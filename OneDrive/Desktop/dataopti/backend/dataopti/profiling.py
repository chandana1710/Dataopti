"""Tabular target inference and detailed, JSON-safe profiling."""

import re

import numpy as np
import pandas as pd

from .detection import detect_dataset_type, detect_sensor_columns
from .quality import score_data_quality
from .validation import inspect_invalid_values
from .value_utils import stable_value


def infer_target_column(df):
    """Infer a likely supervised target from names and column cardinality."""
    if len(df.columns) < 2:
        return {"column": None, "confidence": "none", "reason": "At least two columns are needed to separate a target from features."}

    explicit_terms = {"target", "label", "class", "outcome", "response", "dependent"}
    common_terms = {"churn", "survived", "diagnosis", "species", "status", "result", "category", "rating", "score", "grade", "quality", "price", "sales", "revenue", "profit", "default", "approved", "fraud", "activity"}
    identifier_terms = {"id", "uuid", "guid", "index", "identifier"}
    candidates = []

    for position, (column, series) in enumerate(df.items()):
        values = series.dropna()
        if values.empty:
            continue
        try:
            distinct_values = int(values.nunique())
        except TypeError:
            values = values.map(stable_value)
            distinct_values = int(values.nunique())
        if distinct_values < 2:
            continue
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(column))
        terms = set(re.findall(r"[a-z0-9]+", name.lower()))
        if terms & identifier_terms:
            continue
        explicit = terms & explicit_terms
        if name.strip().lower() == "y":
            explicit.add("y")
        common = terms & common_terms
        numeric = pd.api.types.is_numeric_dtype(series)
        high_cardinality_text = not numeric and distinct_values >= 50 and distinct_values / max(len(values), 1) >= 0.9
        candidates.append({
            "column": str(column),
            "position": position,
            "score": 100 if explicit else 70 if common else 0,
            "distinct_values": distinct_values,
            "high_cardinality_text": high_cardinality_text,
            "explicit_match": bool(explicit),
        })

    if not candidates:
        return {"column": None, "confidence": "none", "reason": "No column with more than one usable value was found."}

    named = [candidate for candidate in candidates if candidate["score"]]
    if named:
        strongest = max(candidate["score"] for candidate in named)
        matches = [candidate for candidate in named if candidate["score"] == strongest]
        selected = max(matches, key=lambda candidate: candidate["position"])
        if len(matches) > 1:
            confidence, reason = "low", "Several columns look like targets; the rightmost strongest match was selected."
        elif selected["explicit_match"]:
            confidence, reason = "high", "The column name explicitly marks it as a target, label, class, or outcome."
        else:
            confidence, reason = "medium", "The column name resembles a common prediction outcome."
    else:
        if detect_sensor_columns(df.columns):
            return {"column": None, "confidence": "none", "reason": "Sensor signals were detected, but no outcome or activity label was found; this dataset will be profiled without supervised training."}
        possible = [candidate for candidate in candidates if not candidate["high_cardinality_text"]]
        if not possible:
            return {"column": None, "confidence": "none", "reason": "The available columns look like identifiers; no likely target was found."}
        selected = possible[-1]
        confidence, reason = "low", "No target-like name was found; the last non-ID column is used as a dataset convention."

    return {"column": selected["column"], "confidence": confidence, "reason": reason, "distinct_values": selected["distinct_values"]}


def _outlier_column_counts(df):
    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric.empty:
        return {}
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    counts = {}
    for column in numeric.columns:
        count = int(((numeric[column] < q1[column] - 1.5 * iqr[column]) | (numeric[column] > q3[column] + 1.5 * iqr[column])).sum())
        if count:
            counts[str(column)] = count
    return counts


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value.item() if hasattr(value, "item") else value


def _sensor_profile(df, sensor_groups):
    timestamp_column = next((
        column for column in df.columns
        if any(term in str(column).lower() for term in ("timestamp", "datetime", "date", "time"))
    ), None)
    timestamp_range = None
    sampling_interval = None
    if timestamp_column is not None:
        timestamps = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True).dropna().sort_values().drop_duplicates()
        if not timestamps.empty:
            timestamp_range = {"start": timestamps.iloc[0].isoformat(), "end": timestamps.iloc[-1].isoformat()}
            intervals = timestamps.diff().dt.total_seconds().dropna()
            if not intervals.empty:
                sampling_interval = round(float(intervals.median()), 6)

    sensor_columns = sorted({column for columns in sensor_groups.values() for column in columns})
    statistics = {}
    for column in sensor_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        statistics[column] = {
            "mean": _json_value(values.mean()),
            "standard_deviation": _json_value(values.std()),
            "minimum": _json_value(values.min()),
            "maximum": _json_value(values.max()),
            "missing_values": int(values.isna().sum()),
        }
    return {
        "timestamp_column": str(timestamp_column) if timestamp_column is not None else None,
        "timestamp_range": timestamp_range,
        "sampling_interval_seconds_median": sampling_interval,
        "sensor_columns": sensor_groups,
        "signal_statistics": statistics,
    }


def analyze_dataframe(df, filename=""):
    """Return summary and per-column statistics for a tabular dataset."""
    rows, columns = df.shape
    missing_by_column = df.isna().sum()
    missing = int(missing_by_column.sum())
    try:
        duplicates = int(df.duplicated().sum())
    except (TypeError, ValueError):
        duplicates = 0
    outliers_by_column = _outlier_column_counts(df)
    outliers = sum(outliers_by_column.values())
    invalid_findings = inspect_invalid_values(df)
    invalid_by_column = {item["column"]: item["count"] for item in invalid_findings}
    invalid_values = sum(invalid_by_column.values())
    profile = []
    numeric_summary = df.describe(include=[np.number]).to_dict() if len(df.select_dtypes(include=[np.number]).columns) else {}

    for column in df.columns:
        series = df[column]
        try:
            unique_count = int(series.nunique(dropna=True))
        except TypeError:
            unique_count = int(series.map(stable_value).nunique(dropna=True))
        missing_count = int(missing_by_column[column])
        item = {
            "name": str(column),
            "dtype": str(series.dtype),
            "kind": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            "missing_count": missing_count,
            "missing_percent": round(100 * missing_count / rows, 2) if rows else 0,
            "unique_count": unique_count,
            "constant": unique_count <= 1,
            "high_cardinality": unique_count > min(50, max(10, int(rows * 0.5))),
            "invalid_count": int(invalid_by_column.get(str(column), 0)),
            "invalid_problem": next((finding["problem"] for finding in invalid_findings if finding["column"] == str(column)), None),
            "invalid_recommended_treatment": next((finding["recommended_treatment"] for finding in invalid_findings if finding["column"] == str(column)), None),
            "recommended_missing_treatment": "drop column (>50% missing)" if rows and missing_count / rows > 0.5 else "replace invalid values, then median" if invalid_by_column.get(str(column)) and pd.api.types.is_numeric_dtype(series) else "median" if pd.api.types.is_numeric_dtype(series) else "most frequent value",
        }
        if pd.api.types.is_numeric_dtype(series):
            item["statistics"] = {
                key: _json_value(numeric_summary.get(column, {}).get(key))
                for key in ("min", "max", "mean", "50%", "std", "25%", "75%")
            }
            item["statistics"].update({"median": item["statistics"].pop("50%"), "q1": item["statistics"].pop("25%"), "q3": item["statistics"].pop("75%")})
            item["outlier_count"] = int(outliers_by_column.get(str(column), 0))
        profile.append(item)

    quality = score_data_quality(df, missing, duplicates, outliers, invalid_values)
    sensors = detect_sensor_columns(df.columns)
    return {
        "rows": int(rows),
        "columns": int(columns),
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "outliers": int(outliers),
        "outliers_by_column": outliers_by_column,
        "invalid_values": invalid_values,
        "invalid_values_by_column": invalid_findings,
        "column_names": [str(column) for column in df.columns],
        "column_profiles": profile,
        "dataset_type": detect_dataset_type(filename, df.columns) if filename else ("sensor" if sensors else "tabular"),
        "sensor_columns": sensors,
        "sensor_profile": _sensor_profile(df, sensors) if sensors else None,
        "data_quality": quality,
        "target_inference": infer_target_column(df),
    }
