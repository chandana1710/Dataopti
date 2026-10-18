"""Helpers for comparing nested values stored in dataframe object columns."""

import json

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def stable_value(value):
    """Return a hashable, deterministic representation of nested cell values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default)
    return value
