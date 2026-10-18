"""Small, type-aware feature engineering steps for tabular and sensor data."""

import re
import pandas.api.types as ptypes

import numpy as np
import pandas as pd

from .detection import detect_sensor_columns


def engineer_features(df, target_name=None):
    """Derive date and three-axis sensor features without using the target."""
    engineered = df.copy()
    created = []
    dropped_source_columns = []
    target_column = next((column for column in engineered.columns if str(column) == target_name), None)

    for column in list(engineered.columns):
        if column == target_column:
            continue
        name = str(column).lower()
        series = engineered[column]
        is_date_name = any(term in name for term in ("date", "time", "timestamp"))
        if not is_date_name and not ptypes.is_datetime64_any_dtype(series):
            continue
        if ptypes.is_numeric_dtype(series):
            median_value = series.dropna().abs().median() if series.notna().any() else 0
            if 10**8 <= median_value < 10**11:
                parsed = pd.to_datetime(series, unit="s", errors="coerce", utc=True)
            elif median_value >= 10**11:
                parsed = pd.to_datetime(series, unit="ms", errors="coerce", utc=True)
            else:
                parsed = pd.to_datetime(series.round().astype("Int64").astype("string"), format="%Y%m%d", errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
        observed = int(series.notna().sum())
        if observed == 0 or parsed.notna().sum() / observed < 0.8:
            continue
        for suffix, values in (
            ("year", parsed.dt.year),
            ("month", parsed.dt.month),
            ("day", parsed.dt.day),
            ("day_of_week", parsed.dt.dayofweek),
            ("hour", parsed.dt.hour),
        ):
            new_name = f"{column}_{suffix}"
            engineered[new_name] = values
            created.append(new_name)
        dropped_source_columns.append(column)

    engineered = engineered.drop(columns=dropped_source_columns, errors="ignore")
    identifier_terms = {"id", "uuid", "guid", "identifier", "index"}
    identifier_columns = [
        column for column in engineered.columns
        if column != target_column
        and set(re.findall(r"[a-z0-9]+", str(column).lower())) & identifier_terms
    ]
    engineered = engineered.drop(columns=identifier_columns, errors="ignore")
    sensors = detect_sensor_columns(engineered.columns)
    for family, columns in sensors.items():
        axis_columns = {}
        for column in columns:
            tokens = set(re.findall(r"[a-z0-9]+", str(column).lower()))
            for axis in ("x", "y", "z"):
                if axis in tokens:
                    axis_columns[axis] = column
        if len(axis_columns) == 3:
            numeric_axes = [pd.to_numeric(engineered[axis_columns[axis]], errors="coerce") for axis in ("x", "y", "z")]
            magnitude_name = f"{family}_magnitude"
            engineered[magnitude_name] = np.sqrt(sum(axis.pow(2) for axis in numeric_axes))
            created.append(magnitude_name)

    return engineered, {
        "generated_features": created,
        "removed_source_columns": [str(column) for column in dropped_source_columns],
        "removed_identifier_columns": [str(column) for column in identifier_columns],
    }
