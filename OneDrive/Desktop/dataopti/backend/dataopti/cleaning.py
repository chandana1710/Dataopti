"""Repeatable dataframe cleaning rules used by upload preparation."""

import re

import numpy as np
import pandas as pd

from .profiling import _outlier_column_counts
from .validation import replace_invalid_values
from .value_utils import stable_value

MAX_ENCODED_CELLS = 10_000_000
MAX_CATEGORIES_PER_COLUMN = 50


def duplicate_mask(df, keep="first"):
    """Find duplicates, including rows containing nested JSON objects."""
    try:
        return df.duplicated(keep=keep)
    except (TypeError, ValueError):
        comparable = df.copy()
        for column in comparable.columns:
            if comparable[column].dtype == "object":
                comparable[column] = comparable[column].map(
                    stable_value
                )
        return comparable.duplicated(keep=keep)


def clean_dataframe(df, target_name=None, make_ml_ready=False):
    """Clean a copy and optionally encode and scale features for ML export."""
    cleaned = df.copy()
    rows_before = len(cleaned)
    missing_values_original = int(cleaned.isna().sum().sum())
    cleaned, invalid_findings = replace_invalid_values(cleaned)
    invalid_values_fixed = sum(item["count"] for item in invalid_findings)
    missing_before = int(cleaned.isna().sum().sum())
    duplicates = duplicate_mask(cleaned, keep="first")
    duplicates_removed = int(duplicates.sum())
    cleaned = cleaned.loc[~duplicates].copy()
    missing_after_deduplication = int(cleaned.isna().sum().sum())

    target_rows_removed = 0
    missing_removed_with_target_rows = 0
    if target_name:
        columns_by_name = {str(column): column for column in cleaned.columns}
        if target_name not in columns_by_name:
            raise ValueError("The automatically detected target column is missing from the dataset.")
        target_column = columns_by_name[target_name]
        missing_target = cleaned[target_column].isna()
        target_rows_removed = int(missing_target.sum())
        missing_removed_with_target_rows = int(cleaned.loc[missing_target].isna().sum().sum())
        cleaned = cleaned.loc[~missing_target].copy()

    empty_columns = [column for column in cleaned.columns if cleaned[column].isna().all()]
    high_missing_columns = [
        column for column in cleaned.columns
        if column != next((c for c in cleaned.columns if str(c) == target_name), None)
        and len(cleaned) and cleaned[column].isna().mean() > 0.5
    ]
    columns_removed = list(dict.fromkeys([*empty_columns, *high_missing_columns]))
    missing_removed_with_removed_columns = int(cleaned[columns_removed].isna().sum().sum()) if columns_removed else 0

    for column in cleaned.columns:
        if column in columns_removed or not cleaned[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            fill_value = cleaned[column].median()
        else:
            try:
                modes = cleaned[column].mode(dropna=True)
            except (TypeError, ValueError):
                modes = pd.Series(dtype="object")
            fill_value = modes.iloc[0] if not modes.empty else "Unknown"
            cleaned[column] = cleaned[column].astype("object")
        if fill_value is None or (np.isscalar(fill_value) and pd.isna(fill_value)):
            fill_value = 0 if pd.api.types.is_numeric_dtype(cleaned[column]) else "Unknown"
        cleaned[column] = cleaned[column].fillna(fill_value)

    outliers_by_column = _outlier_column_counts(cleaned)
    outliers_detected = sum(outliers_by_column.values())
    if columns_removed:
        cleaned = cleaned.drop(columns=columns_removed)
    if target_name and target_name not in {str(column) for column in cleaned.columns}:
        raise ValueError("The automatically detected target column is empty or could not be retained.")

    encoded_columns, scaled_columns = [], []
    preprocessing_limited = False
    estimated_encoded_cells = 0
    if make_ml_ready:
        target_column = next(column for column in cleaned.columns if str(column) == target_name)
        target_values = cleaned[target_column]
        features = cleaned.drop(columns=[target_column])
        numeric_columns = [column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])]
        id_terms = {"id", "uuid", "guid", "identifier", "index"}
        scale_columns = [column for column in numeric_columns if not (set(re.findall(r"[a-z0-9]+", str(column).lower())) & id_terms)]
        categorical_columns = [column for column in features.columns if column not in numeric_columns]
        encoded_columns = [str(column) for column in categorical_columns]
        scaled_columns = [str(column) for column in scale_columns]
        for column in categorical_columns:
            features[column] = features[column].map(stable_value)

        # Bound category width before expanding it. Embedding arrays and free-text
        # columns often have one distinct value per row and can otherwise create
        # a multi-gigabyte dense dummy matrix.
        category_counts = {}
        for column in categorical_columns:
            counts = features[column].value_counts(dropna=False)
            if len(counts) > MAX_CATEGORIES_PER_COLUMN:
                retained = set(counts.nlargest(MAX_CATEGORIES_PER_COLUMN - 1).index)
                features[column] = features[column].where(features[column].isin(retained), "__OTHER__")
            category_counts[column] = int(features[column].nunique(dropna=False))

        estimated_encoded_cells = len(features) * (
            len(numeric_columns) + sum(category_counts.values())
        )
        preprocessing_limited = estimated_encoded_cells > MAX_ENCODED_CELLS
        if not preprocessing_limited:
            features = pd.get_dummies(features, columns=categorical_columns, dtype=float)
        else:
            encoded_columns = []
        for column in scale_columns:
            deviation = features[column].std(ddof=0)
            features[column] = (features[column] - features[column].mean()) / deviation if pd.notna(deviation) and deviation > 0 else 0.0
        if not pd.api.types.is_numeric_dtype(target_values):
            target_values, _ = pd.factorize(target_values.map(stable_value), sort=True)
        target_frame = pd.DataFrame({str(target_name): target_values}, index=cleaned.index)
        cleaned = pd.concat([features, target_frame], axis=1)

    missing_after = int(cleaned.isna().sum().sum())
    report = {
        "rows_before": int(rows_before),
        "rows_after": int(len(cleaned)),
        "duplicate_rows_removed": duplicates_removed,
        "rows_without_target_removed": target_rows_removed,
        "missing_values_before": missing_before,
        "missing_values_original": missing_values_original,
        "missing_values_filled": max(0, missing_after_deduplication - missing_removed_with_target_rows - missing_removed_with_removed_columns - missing_after),
        "missing_values_after": missing_after,
        "missing_values_removed_with_duplicate_rows": missing_before - missing_after_deduplication,
        "all_empty_columns_removed": [str(column) for column in empty_columns],
        "high_missing_columns_removed": [str(column) for column in high_missing_columns],
        "invalid_values_treated": invalid_values_fixed,
        "invalid_values_by_column": invalid_findings,
        "outliers_detected": int(outliers_detected),
        "outliers_by_column": outliers_by_column,
        "outliers_removed": 0,
        "encoded_columns": encoded_columns,
        "scaled_columns": scaled_columns,
        "categorical_encoding_skipped_for_size": preprocessing_limited,
        "estimated_encoded_cells": int(estimated_encoded_cells),
        "cleaning_rules": [
            "Remove exact duplicate rows.",
            "Remove rows with a missing automatically detected target.",
            "Fill numeric gaps with the median and categorical gaps with the mode.",
            "Drop empty columns and non-target columns with more than 50% missing values.",
            "Detect and report numeric IQR outliers; keep them unchanged.",
            "Encode categorical features up to a bounded category count and standardize numeric features for ML export.",
        ],
    }
    return cleaned, report
