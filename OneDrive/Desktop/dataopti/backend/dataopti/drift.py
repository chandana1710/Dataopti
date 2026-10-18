"""Compare two dataset versions and estimate per-column distribution drift."""

import numpy as np
import pandas as pd


def _level(score, moderate, high):
    if score >= high:
        return "high"
    if score >= moderate:
        return "moderate"
    return "low"


def _numeric_drift(reference, current):
    reference = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    current = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if reference.empty or current.empty:
        return {"method": "numeric population stability index", "score": None, "status": "unavailable", "reason": "A dataset version has no usable numeric values."}
    edges = np.unique(reference.quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float))
    if len(edges) < 2:
        return {"method": "numeric population stability index", "score": 0.0, "status": "low", "reason": "The reference column is constant."}
    edges[0], edges[-1] = -np.inf, np.inf
    expected = pd.cut(reference, bins=edges, include_lowest=True).value_counts(sort=False, normalize=True).to_numpy(dtype=float)
    actual = pd.cut(current, bins=edges, include_lowest=True).value_counts(sort=False, normalize=True).to_numpy(dtype=float)
    epsilon = 1e-6
    score = float(np.sum((actual - expected) * np.log((actual + epsilon) / (expected + epsilon))))
    return {"method": "numeric population stability index", "score": round(score, 6), "status": _level(score, 0.1, 0.25), "reference_non_missing": int(len(reference)), "current_non_missing": int(len(current))}


def _categorical_drift(reference, current):
    reference = reference.astype("string").fillna("<missing>").value_counts(normalize=True)
    current = current.astype("string").fillna("<missing>").value_counts(normalize=True)
    categories = reference.index.union(current.index)
    score = float((reference.reindex(categories, fill_value=0) - current.reindex(categories, fill_value=0)).abs().sum() / 2)
    return {"method": "categorical total variation distance", "score": round(score, 6), "status": _level(score, 0.1, 0.25), "category_count": int(len(categories))}


def compare_datasets(reference, current):
    """Return measurable drift for columns shared by both dataframe versions."""
    shared = [column for column in reference.columns if column in current.columns]
    if not shared:
        raise ValueError("The reference and current dataset have no columns in common.")

    columns = []
    for column in shared:
        reference_missing_rate = float(reference[column].isna().mean()) if len(reference) else 0.0
        current_missing_rate = float(current[column].isna().mean()) if len(current) else 0.0
        missing_rate_change = abs(current_missing_rate - reference_missing_rate)
        if pd.api.types.is_numeric_dtype(reference[column]) and pd.api.types.is_numeric_dtype(current[column]):
            result = _numeric_drift(reference[column], current[column])
        else:
            result = _categorical_drift(reference[column], current[column])
        missing_status = _level(missing_rate_change, 0.1, 0.25)
        if result["status"] == "unavailable" and missing_rate_change >= 0.1:
            result["status"] = missing_status
        elif result["status"] != "unavailable" and {"low": 0, "moderate": 1, "high": 2}[missing_status] > {"low": 0, "moderate": 1, "high": 2}[result["status"]]:
            result["status"] = missing_status
        result["reference_missing_rate"] = round(reference_missing_rate, 6)
        result["current_missing_rate"] = round(current_missing_rate, 6)
        result["missing_rate_change"] = round(missing_rate_change, 6)
        columns.append({"column": str(column), **result})

    added_columns = [str(column) for column in current.columns if column not in reference.columns]
    removed_columns = [str(column) for column in reference.columns if column not in current.columns]
    levels = [item["status"] for item in columns if item["status"] != "unavailable"]
    structural_change = bool(added_columns or removed_columns)
    overall = "unavailable" if not levels else "high" if "high" in levels else "moderate" if "moderate" in levels else "low"
    if structural_change and overall in {"low", "unavailable"}:
        overall = "moderate"
    changed = [item["column"] for item in columns if item["status"] in {"moderate", "high"}]
    changed.extend(f"Added: {column}" for column in added_columns)
    changed.extend(f"Removed: {column}" for column in removed_columns)
    return {
        "status": overall,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "shared_columns": len(shared),
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "columns": columns,
        "drifted_columns": changed,
        "retraining_recommended": overall in {"moderate", "high"},
        "recommendation": "Evaluate the model again with the current dataset." if overall in {"moderate", "high"} else "No material distribution shift was measured; continue monitoring.",
    }
