"""Conservative validation rules for clearly impossible numeric values."""

import numpy as np
import pandas as pd


def invalid_mask(series, column_name):
    """Return a mask for invalid values only when column semantics are clear."""
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index)
    name = str(column_name).lower().replace("-", "_").replace(" ", "_")
    values = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)), index=series.index)

    if "latitude" in name or name in {"lat", "gps_lat"}:
        return mask | (values < -90) | (values > 90)
    if "longitude" in name or name in {"lon", "lng", "gps_lon"}:
        return mask | (values < -180) | (values > 180)
    if "age" in name:
        upper = 1200 if "month" in name else 120
        return mask | (values < 0) | (values > upper)
    if any(term in name for term in ("percent", "percentage", "pct")):
        return mask | (values < 0) | (values > 100)
    if any(term in name for term in ("salary", "income", "revenue", "sales", "price", "cost", "wage")):
        return mask | (values < 0)
    if any(term in name for term in ("count", "quantity", "qty", "units")):
        return mask | (values < 0)
    return mask


def inspect_invalid_values(df):
    findings = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mask = invalid_mask(df[column], column)
            problem = _problem_for_column(column)
            treatment = "Replace impossible values with missing values, then apply the column's imputation rule."
        else:
            mask = df[column].map(lambda value: isinstance(value, str) and value != value.strip())
            problem = "Text values contain leading or trailing whitespace."
            treatment = "Trim leading and trailing whitespace from text values."
        count = int(mask.sum())
        if not count:
            continue
        findings.append({
            "column": str(column),
            "count": count,
            "examples": [value.item() if hasattr(value, "item") else value for value in df.loc[mask, column].head(5)],
            "problem": problem,
            "recommended_treatment": treatment,
        })
    return findings


def replace_invalid_values(df):
    cleaned = df.copy()
    findings = inspect_invalid_values(cleaned)
    for finding in findings:
        column = next(column for column in cleaned.columns if str(column) == finding["column"])
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned.loc[invalid_mask(cleaned[column], column), column] = np.nan
        else:
            cleaned[column] = cleaned[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return cleaned, findings


def _problem_for_column(column):
    name = str(column).lower()
    if "age" in name:
        return "Age is outside the supported range."
    if "latitude" in name or name in {"lat", "gps_lat"}:
        return "Latitude must be between -90 and 90 degrees."
    if "longitude" in name or name in {"lon", "lng", "gps_lon"}:
        return "Longitude must be between -180 and 180 degrees."
    if any(term in name for term in ("percent", "percentage", "pct")):
        return "Percentage is outside the 0 to 100 range."
    return "A non-negative quantity contains a negative value."
